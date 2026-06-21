"""Train a scratch CNN on DAMP-S-AG log-mel spectrograms."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Sandbox.singerclassifier.data import build_dataloader
from Sandbox.singerclassifier.experiments import (
    augmentation_diagnostics_from_config,
    dataloader_kwargs_from_config,
    print_augmentation_diagnostics,
)
from Sandbox.singerclassifier.metrics import (
    save_classification_report,
    save_confusion_matrix_plot,
)
from Sandbox.singerclassifier.models import build_model
from Sandbox.singerclassifier.train_utils import (
    DEFAULT_CLASS_NAMES,
    EarlyStopping,
    compute_class_weights_from_split_csv,
    evaluate_one_epoch,
    get_device,
    get_git_commit_hash,
    load_yaml,
    run_multicrop_inference,
    save_json,
    save_yaml,
    set_seed,
    train_one_epoch,
)


def merge_config_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if args.run_name:
        resolved["run_name"] = args.run_name
    if args.epochs is not None:
        resolved.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        resolved.setdefault("data", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        resolved.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.device:
        resolved["device"] = args.device
    return resolved


def build_dataloaders(config: dict[str, Any]) -> tuple[Any, Any]:
    train_loader = build_dataloader(**dataloader_kwargs_from_config(config, "train"))
    val_loader = build_dataloader(**dataloader_kwargs_from_config(config, "val"))
    return train_loader, val_loader


def evaluate_validation(
    model: nn.Module,
    config: dict[str, Any],
    val_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    eval_num_crops = int(config.get("data", {}).get("eval_num_crops", 1))
    if eval_num_crops > 1:
        return run_multicrop_inference(
            model=model,
            config=config,
            split="val",
            device=device,
            class_names=class_names,
        )
    return evaluate_one_epoch(model, val_loader, criterion, device, class_names)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: dict[str, Any],
    class_names: list[str],
    best_epoch: int,
    best_val_macro_f1: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "class_names": class_names,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
        },
        path,
    )


def train(config: dict[str, Any]) -> Path:
    seed = config.get("seed", 42)
    set_seed(seed)

    device = get_device(config.get("device"))
    class_names = config.get("class_names", DEFAULT_CLASS_NAMES)

    run_name = config["run_name"]
    run_dir = Path(config["output"]["root_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = copy.deepcopy(config)
    resolved_config["class_names"] = class_names
    git_hash = get_git_commit_hash()
    if git_hash:
        resolved_config["git_commit"] = git_hash

    diag = print_augmentation_diagnostics(resolved_config)
    resolved_config["augmentation_diagnostics"] = diag
    save_yaml(resolved_config, run_dir / "config.yaml")

    train_loader, val_loader = build_dataloaders(resolved_config)

    model_cfg = resolved_config["model"]
    model = build_model(
        model_name=model_cfg.get("name", "small_mel_cnn"),
        num_classes=model_cfg.get("num_classes", 3),
        dropout=model_cfg.get("dropout", 0.25),
        classifier_dropout=model_cfg.get("classifier_dropout", 0.4),
    ).to(device)

    train_cfg = resolved_config["training"]
    class_weights = None
    if train_cfg.get("class_weighted_loss", False):
        class_weights = compute_class_weights_from_split_csv(
            resolved_config["data"]["split_csv"],
            split="train",
            num_classes=model_cfg.get("num_classes", 3),
        ).to(device)
        print(f"Class weights: {class_weights.tolist()}")
        resolved_config["class_weights"] = class_weights.detach().cpu().tolist()
        save_yaml(resolved_config, run_dir / "config.yaml")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    early_stopping = EarlyStopping(
        patience=train_cfg.get("early_stopping_patience", 7)
    )

    history: list[dict[str, Any]] = []
    best_state = None
    epochs = train_cfg.get("epochs", 30)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, class_names
        )
        val_metrics = evaluate_validation(
            model, resolved_config, val_loader, criterion, device, class_names
        )

        lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "learning_rate": lr,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"train_acc={epoch_record['train_accuracy']:.4f} "
            f"train_macro_f1={epoch_record['train_macro_f1']:.4f} | "
            f"val_loss={epoch_record['val_loss']:.4f} "
            f"val_acc={epoch_record['val_accuracy']:.4f} "
            f"val_macro_f1={epoch_record['val_macro_f1']:.4f} "
            f"val_balanced_acc={epoch_record['val_balanced_accuracy']:.4f} "
            f"lr={lr:.6f}"
        )

        improved = early_stopping.step(
            metric=val_metrics["macro_f1"],
            loss=val_metrics["loss"],
            epoch=epoch,
        )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                run_dir / "best_model.pt",
                model,
                resolved_config,
                class_names,
                epoch,
                val_metrics["macro_f1"],
            )

        save_checkpoint(
            run_dir / "last_model.pt",
            model,
            resolved_config,
            class_names,
            epoch,
            val_metrics["macro_f1"],
        )

        if early_stopping.should_stop:
            print(
                f"Early stopping triggered after epoch {epoch} "
                f"(best epoch={early_stopping.best_epoch}, "
                f"best val_macro_f1={early_stopping.best_metric:.4f})"
            )
            break

    save_json({"history": history}, run_dir / "history.json")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_metrics = evaluate_validation(
        model, resolved_config, val_loader, criterion, device, class_names
    )
    save_json(final_val_metrics, run_dir / "val_metrics.json")
    save_classification_report(
        final_val_metrics,
        run_dir / "val_classification_report.txt",
    )
    save_confusion_matrix_plot(
        final_val_metrics["confusion_matrix"],
        class_names,
        run_dir / "val_confusion_matrix.png",
    )

    print(f"\nTraining complete. Run directory: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SmallMelCNN")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    config = merge_config_overrides(config, args)
    train(config)


if __name__ == "__main__":
    main()
