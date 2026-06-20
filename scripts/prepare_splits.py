"""Prepare leakage-safe train/val/test splits for DAMP-S-AG."""

from __future__ import annotations

import argparse
from pathlib import Path

from Sandbox.singerclassifier.data import (
    DEFAULT_BUCKET_NAMES,
    DEFAULT_BUCKET_THRESHOLDS,
    assign_age_buckets,
    load_metadata,
)
from Sandbox.singerclassifier.splits import (
    create_splits,
    print_split_summary,
    save_splits,
    validate_split_ratios,
)
from Sandbox.singerclassifier.utils import SPLIT_CSV_PATH, SPLIT_SUMMARY_JSON_PATH


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
        default=SPLIT_CSV_PATH,
        help="Output path for split CSV",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=SPLIT_SUMMARY_JSON_PATH,
        help="Output path for split summary JSON",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-age", type=int, default=10)
    parser.add_argument("--max-age", type=int, default=90)
    args = parser.parse_args()

    validate_split_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    print(f"Loading metadata from: {args.data_root}")
    df = load_metadata(args.data_root, min_age=args.min_age, max_age=args.max_age)
    print(f"Rows after filtering: {len(df)}")

    df, class_mapping = assign_age_buckets(
        df,
        thresholds=DEFAULT_BUCKET_THRESHOLDS,
        bucket_names=DEFAULT_BUCKET_NAMES,
    )

    split_df, summary = create_splits(
        df,
        class_mapping=class_mapping,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_csv=args.output_csv,
    )

    save_splits(
        split_df,
        output_csv=args.output_csv,
        summary=summary,
        summary_json=args.summary_json,
    )

    print_split_summary(summary, args.output_csv, args.summary_json)
    print(f"\nSaved split CSV to: {args.output_csv}")
    print(f"Saved split summary to: {args.summary_json}")


if __name__ == "__main__":
    main()
