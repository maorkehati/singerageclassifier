"""Prepare leakage-safe train/val/test splits for DAMP-S-AG."""

from __future__ import annotations

import argparse
from pathlib import Path

from Sandbox.singerclassifier.data import assign_age_buckets, load_metadata
from Sandbox.singerclassifier.splits import create_splits, save_splits
from Sandbox.singerclassifier.utils import print_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DAMP-S-AG train/val/test splits"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to DAMP-S-AG folder containing amazing_grace.tsv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/damp_sag_splits.csv"),
        help="Output path for split CSV",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("experiments/data_inspection/split_summary.json"),
        help="Output path for split summary JSON",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bucket-threshold-low",
        type=int,
        default=25,
        help="Upper bound (exclusive) for under_25 bucket",
    )
    parser.add_argument(
        "--bucket-threshold-high",
        type=int,
        default=35,
        help="Upper bound (exclusive) for age_25_34 bucket",
    )
    parser.add_argument("--min-age", type=int, default=10)
    parser.add_argument("--max-age", type=int, default=90)
    args = parser.parse_args()

    print(f"Loading metadata from: {args.data_root}")
    df = load_metadata(args.data_root, min_age=args.min_age, max_age=args.max_age)
    print(f"Rows after filtering: {len(df)}")

    df, class_mapping = assign_age_buckets(
        df,
        thresholds=(args.bucket_threshold_low, args.bucket_threshold_high),
    )

    split_df, summary = create_splits(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    summary["class_mapping"] = class_mapping
    save_splits(
        split_df,
        output_csv=args.output_csv,
        summary=summary,
        summary_json=args.summary_json,
        class_mapping=class_mapping,
    )

    print_table("Final row counts", summary["rows_per_split"])
    print_table("Final account counts", summary["accounts_per_split"])


if __name__ == "__main__":
    main()
