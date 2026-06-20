"""Run the majority-class baseline experiment."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from Sandbox.singerclassifier.experiments import get_run_dir
from Sandbox.singerclassifier.metrics import (
    compute_classification_metrics,
    save_classification_report,
    save_confusion_matrix_plot,
)
from Sandbox.singerclassifier.train_utils import (
    DEFAULT_CLASS_NAMES,
    get_git_commit_hash,
    load_yaml,
    save_json,
    save_yaml,
    set_seed,
)


def run_majority_baseline(config: dict) -> Path:
    seed = config.get("seed", 42)
    set_seed(seed)

    split_csv = config["data"]["split_csv"]
    df = pd.read_csv(split_csv)
    train_df = df[df["split"] == "train"]
    if train_df.empty:
        raise ValueError("Train split is empty in split CSV.")

    majority_class = int(train_df["age_bucket_id"].mode().iloc[0])
    class_names = config.get("class_names", DEFAULT_CLASS_NAMES)

    run_dir = get_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = copy.deepcopy(config)
    resolved_config["class_names"] = class_names
    resolved_config["majority_class"] = majority_class
    git_hash = get_git_commit_hash()
    if git_hash:
        resolved_config["git_commit"] = git_hash
    save_yaml(resolved_config, run_dir / "config.yaml")

    print(f"Majority class from train split: {majority_class}")

    for split in ("val", "test"):
        split_df = df[df["split"] == split].reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"Split '{split}' is empty in split CSV.")

        y_true = split_df["age_bucket_id"].astype(int).tolist()
        y_pred = [majority_class] * len(y_true)
        metrics = compute_classification_metrics(
            y_true,
            y_pred,
            class_names=class_names,
        )
        metrics["loss"] = 0.0
        metrics["majority_class"] = majority_class

        save_json(metrics, run_dir / f"{split}_metrics.json")
        save_classification_report(metrics, run_dir / f"{split}_classification_report.txt")
        save_confusion_matrix_plot(
            metrics["confusion_matrix"],
            class_names,
            run_dir / f"{split}_confusion_matrix.png",
        )

        print(
            f"{split}: accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
        )

    print(f"\nMajority baseline complete. Run directory: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run majority-class baseline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    run_majority_baseline(config)


if __name__ == "__main__":
    main()
