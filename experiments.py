"""Controlled experiment definitions and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sweep import resolve_augmentation_cfg

PHASE6_CONFIGS = [
    "configs/majority_baseline.yaml",
    "configs/cnn_basic.yaml",
    "configs/cnn_balanced.yaml",
    "configs/cnn_augmented.yaml",
    "configs/cnn_augmented_multicrop.yaml",
]

EXPERIMENT_METADATA: dict[str, dict[str, str]] = {
    "majority_baseline": {
        "main_change": "Predict most frequent train class",
        "notes": "Trivial reference point",
    },
    "cnn_basic": {
        "main_change": "Scratch CNN over log-mel spectrograms",
        "notes": "First valid deep-learning model",
    },
    "cnn_balanced": {
        "main_change": "+ class-weighted cross entropy",
        "notes": "Tests imbalance-aware training",
    },
    "cnn_augmented": {
        "main_change": "+ light/medium augmentation",
        "notes": "Tests generalization from augmentation",
    },
    "cnn_augmented_multicrop": {
        "main_change": "+ multi-crop evaluation",
        "notes": "Tests recording-level prediction stability",
    },
}


def resolve_config_path(config_path: str | Path, repo_root: Path | None = None) -> Path:
    path = Path(config_path)
    if path.is_file():
        return path
    if repo_root is not None:
        candidate = repo_root / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Config not found: {config_path}")


def get_run_dir(config: dict[str, Any]) -> Path:
    return Path(config["output"]["root_dir"]) / config["run_name"]


def is_majority_baseline_config(config: dict[str, Any]) -> bool:
    return config.get("model", {}).get("type") == "majority_class"


def dataloader_kwargs_from_config(
    config: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    """Build kwargs for build_dataloader from an experiment config."""
    data_cfg = config["data"]
    aug_cfg = resolve_augmentation_cfg(config)

    if split == "train":
        random_crop = data_cfg.get("random_crop_train", True)
    else:
        random_crop = False

    return {
        "split_csv": data_cfg["split_csv"],
        "split": split,
        "batch_size": data_cfg.get("batch_size", 32),
        "num_workers": data_cfg.get("num_workers", 2),
        "sample_rate": data_cfg.get("sample_rate", 22050),
        "duration_sec": data_cfg.get("duration_sec", 15.0),
        "n_fft": data_cfg.get("n_fft", 1024),
        "hop_length": data_cfg.get("hop_length", 512),
        "n_mels": data_cfg.get("n_mels", 80),
        "f_min": data_cfg.get("f_min", 50.0),
        "f_max": data_cfg.get("f_max", 8000.0),
        "random_crop": random_crop,
        "augment_train": aug_cfg is not None and split == "train",
        "augmentation_cfg": aug_cfg,
    }


def test_metrics_path(config: dict[str, Any]) -> Path:
    return get_run_dir(config) / "test_metrics.json"
