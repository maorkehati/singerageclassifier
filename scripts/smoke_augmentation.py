"""Smoke tests for train-only augmentation wiring."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch

from Sandbox.singerclassifier.data import DampSAGDataset
from Sandbox.singerclassifier.experiments import (
    augmentation_diagnostics_from_config,
    dataloader_kwargs_from_config,
    print_augmentation_diagnostics,
)
from Sandbox.singerclassifier.sweep import resolve_augmentation_cfg
from Sandbox.singerclassifier.train_utils import load_yaml


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def smoke_resolve_augmentation_profile() -> None:
    """Profile-based configs must not be disabled by sweep defaults."""
    config = {
        "data": {"augment_train": True},
        "augmentation": {"enabled": False, "profile": "light"},
    }
    aug_cfg = resolve_augmentation_cfg(config)
    if aug_cfg is None:
        raise AssertionError("Expected resolved augmentation config, got None")
    if not aug_cfg.get("enabled", False):
        raise AssertionError(
            f"Profile light should enable augmentation, got: {aug_cfg}"
        )
    if float(aug_cfg.get("waveform_noise_std", 0.0)) <= 0.0:
        raise AssertionError(
            f"Profile light should set waveform_noise_std, got: {aug_cfg}"
        )
    print("resolve_augmentation_cfg profile merge: OK")


def _dataset_from_config(
    config: dict[str, Any],
    split: str,
    *,
    random_crop: bool = False,
) -> DampSAGDataset:
    kwargs = dataloader_kwargs_from_config(config, split)
    kwargs["random_crop"] = random_crop
    return DampSAGDataset(**kwargs)


def smoke_config_augmentation(
    config: dict[str, Any],
    sample_index: int = 0,
    balanced_config: dict[str, Any] | None = None,
) -> None:
    """Verify augmentation changes train samples but not validation."""
    diag = augmentation_diagnostics_from_config(config)
    print_augmentation_diagnostics(config)

    if not diag["augment_train"]:
        raise AssertionError(
            "Config must have augment_train=true for augmented smoke test"
        )
    if diag["resolved_augmentation"] is None:
        raise AssertionError(
            "augment_train=true but resolved_augmentation is None; "
            "augmentation is a silent no-op"
        )

    train_ds = _dataset_from_config(config, "train", random_crop=False)
    val_ds = _dataset_from_config(config, "val", random_crop=False)

    if not train_ds.augment_train:
        raise AssertionError("Train dataset must have augment_train=True")
    if train_ds.spec_augment is None and train_ds.waveform_noise_std <= 0.0:
        raise AssertionError(
            "Train dataset has no active augmentation operations"
        )
    if val_ds.augment_train or val_ds.spec_augment is not None:
        raise AssertionError("Validation dataset must not apply augmentation")

    torch.manual_seed(0)
    train_a = train_ds[sample_index][0]
    torch.manual_seed(1)
    train_b = train_ds[sample_index][0]
    train_same_seed_a = train_ds[sample_index][0]
    torch.manual_seed(0)
    train_same_seed_b = train_ds[sample_index][0]

    train_diff = mean_abs_diff(train_a, train_b)
    train_same_seed_diff = mean_abs_diff(train_same_seed_a, train_same_seed_b)

    val_a = val_ds[sample_index][0]
    val_b = val_ds[sample_index][0]
    val_diff = mean_abs_diff(val_a, val_b)

    print(f"train same-index mean abs diff (seed 0 vs 1): {train_diff:.6f}")
    print(f"train same-index mean abs diff (seed 0 vs 0): {train_same_seed_diff:.6f}")
    print(f"val same-index mean abs diff: {val_diff:.6f}")

    if train_diff <= 0.0:
        raise AssertionError(
            "Augmented train samples were identical across seeds; "
            "augmentation is not changing training inputs"
        )
    if val_diff != 0.0 and not torch.allclose(val_a, val_b):
        raise AssertionError(
            f"Validation sample changed between reads (mean abs diff={val_diff})"
        )

    if balanced_config is not None:
        balanced_train = _dataset_from_config(
            balanced_config, "train", random_crop=False
        )
        if balanced_train.augment_train:
            raise AssertionError("Balanced config must not enable augment_train")

        torch.manual_seed(0)
        balanced_a = balanced_train[sample_index][0]
        torch.manual_seed(1)
        balanced_b = balanced_train[sample_index][0]
        balanced_diff = mean_abs_diff(balanced_a, balanced_b)
        print(f"balanced train same-index mean abs diff (seed 0 vs 1): {balanced_diff:.6f}")
        if balanced_diff > 0.0:
            raise AssertionError(
                "Balanced train samples differed without augmentation; "
                "disable random crop for this comparison"
            )

    print(f"config augmentation wiring (sample_index={sample_index}): OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test augmentation wiring")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Generated or hand-written experiment config (augmented run)",
    )
    parser.add_argument(
        "--balanced-config",
        type=Path,
        default=None,
        help="Optional matching non-augmented config for comparison",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Legacy path override; normally taken from --config",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Train/val row index to compare",
    )
    args = parser.parse_args()

    smoke_resolve_augmentation_profile()

    if args.config is None:
        print("No --config provided; skipping dataset-level checks.")
        print("\nAugmentation smoke test passed (config resolution only).")
        return

    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = load_yaml(args.config)
    if args.split_csv is not None:
        config = copy.deepcopy(config)
        config.setdefault("data", {})["split_csv"] = str(args.split_csv)

    balanced_config = None
    if args.balanced_config is not None:
        if not args.balanced_config.is_file():
            raise FileNotFoundError(
                f"Balanced config not found: {args.balanced_config}"
            )
        balanced_config = load_yaml(args.balanced_config)

    smoke_config_augmentation(
        config,
        sample_index=args.sample_index,
        balanced_config=balanced_config,
    )

    print("\nAugmentation smoke test passed.")


if __name__ == "__main__":
    main()
