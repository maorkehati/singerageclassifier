"""Evaluate a saved training run on a chosen split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Sandbox.singerclassifier.data import build_dataloader
from Sandbox.singerclassifier.experiments import dataloader_kwargs_from_config
from Sandbox.singerclassifier.metrics import (
    save_classification_report,
    save_confusion_matrix_plot,
)
from Sandbox.singerclassifier.models import build_model
from Sandbox.singerclassifier.train_utils import (
    DEFAULT_CLASS_NAMES,
    evaluate_one_epoch,
    get_device,
    load_yaml,
    run_multicrop_inference,
    save_json,
)


def evaluate_run(
    run_dir: Path,
    split: str = "test",
    checkpoint: str = "best_model.pt",
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and evaluate on the requested split."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_yaml(config_path)
    device = device or get_device(config.get("device"))
    class_names = config.get("class_names", DEFAULT_CLASS_NAMES)

    checkpoint_path = run_dir / checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    model_cfg = config["model"]
    model = build_model(
        model_name=model_cfg.get("name", "small_mel_cnn"),
        num_classes=model_cfg.get("num_classes", 3),
        dropout=model_cfg.get("dropout", 0.25),
        classifier_dropout=model_cfg.get("classifier_dropout", 0.4),
    ).to(device)
    model.load_state_dict(checkpoint_obj["model_state_dict"])

    data_cfg = config["data"]
    eval_num_crops = int(data_cfg.get("eval_num_crops", 1))
    if eval_num_crops > 1:
        metrics = run_multicrop_inference(
            model=model,
            config=config,
            split=split,
            device=device,
            class_names=class_names,
        )
    else:
        loader = build_dataloader(**dataloader_kwargs_from_config(config, split))
        criterion = nn.CrossEntropyLoss()
        metrics = evaluate_one_epoch(model, loader, criterion, device, class_names)

    metrics_path = run_dir / f"{split}_metrics.json"
    report_path = run_dir / f"{split}_classification_report.txt"
    cm_path = run_dir / f"{split}_confusion_matrix.png"

    save_json(metrics, metrics_path)
    save_classification_report(metrics, report_path)
    save_confusion_matrix_plot(metrics["confusion_matrix"], class_names, cm_path)

    print("\nEvaluation summary")
    print("------------------")
    print(f"split: {split}")
    print(f"loss: {metrics['loss']:.4f}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"\nSaved metrics to: {metrics_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename inside run-dir",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    evaluate_run(
        run_dir=args.run_dir,
        split=args.split,
        checkpoint=args.checkpoint,
        device=get_device(args.device) if args.device else None,
    )


if __name__ == "__main__":
    main()
