"""Smoke test for multi-crop evaluation device placement."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from Sandbox.singerclassifier.features import (
    LogMelSpectrogram,
    assert_waveform_feature_extractor_same_device,
)
from Sandbox.singerclassifier.models import build_model
from Sandbox.singerclassifier.train_utils import (
    DEFAULT_CLASS_NAMES,
    get_device,
    load_yaml,
    run_multicrop_inference,
)


def smoke_feature_extractor_device(device: torch.device) -> None:
    """Verify stacked crops and LogMelSpectrogram share a device."""
    feature_extractor = LogMelSpectrogram().to(device)
    waveforms = torch.randn(3, 1, 22050, device=device)
    assert_waveform_feature_extractor_same_device(waveforms, feature_extractor)
    log_mels = feature_extractor(waveforms)
    if log_mels.device != device:
        raise AssertionError(
            f"Expected log-mels on {device}, got {log_mels.device}"
        )
    print(f"feature extractor device check ({device}): OK")


def smoke_multicrop_eval(
    config: dict,
    split: str,
    device: torch.device,
    max_samples: int,
    checkpoint_path: Path | None,
) -> None:
    """Run a short multi-crop evaluation pass."""
    model_cfg = config.get("model", {})
    model = build_model(
        model_name=model_cfg.get("name", "small_mel_cnn"),
        num_classes=model_cfg.get("num_classes", 3),
        dropout=model_cfg.get("dropout", 0.25),
        classifier_dropout=model_cfg.get("classifier_dropout", 0.4),
    ).to(device)

    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint_obj = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_obj["model_state_dict"])
    else:
        print("No checkpoint loaded; using random model weights for smoke test.")

    class_names = config.get("class_names", DEFAULT_CLASS_NAMES)
    metrics = run_multicrop_inference(
        model=model,
        config=config,
        split=split,
        device=device,
        class_names=class_names,
        max_samples=max_samples,
    )

    print("\nMulti-crop smoke metrics")
    print("------------------------")
    print(f"split: {split}")
    print(f"samples evaluated: {max_samples}")
    print(f"eval_num_crops: {metrics.get('eval_num_crops')}")
    print(f"loss: {metrics['loss']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test multi-crop evaluation device placement"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Generated multi-crop experiment config YAML",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Saved run directory containing config.yaml",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Number of recordings to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename inside run-dir (ignored with --config only)",
    )
    args = parser.parse_args()

    if args.config is None and args.run_dir is None:
        raise ValueError("Provide --config or --run-dir")

    device = get_device(args.device)
    smoke_feature_extractor_device(device)

    if args.run_dir is not None:
        config_path = args.run_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        config = load_yaml(config_path)
        checkpoint_path = args.run_dir / args.checkpoint
    else:
        if not args.config.is_file():
            raise FileNotFoundError(f"Config not found: {args.config}")
        config = load_yaml(args.config)
        checkpoint_path = None

    eval_num_crops = int(config.get("data", {}).get("eval_num_crops", 1))
    if eval_num_crops <= 1:
        raise ValueError(
            f"Config must set data.eval_num_crops > 1, got {eval_num_crops}"
        )

    smoke_multicrop_eval(
        config=config,
        split=args.split,
        device=device,
        max_samples=args.max_samples,
        checkpoint_path=checkpoint_path,
    )

    print("\nMulti-crop evaluation smoke test passed.")


if __name__ == "__main__":
    main()
