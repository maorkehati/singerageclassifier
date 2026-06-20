"""Inspect DAMP-S-AG metadata and audio availability."""

from __future__ import annotations

import argparse
from pathlib import Path

from Sandbox.singerclassifier.data import (
    AUDIO_SUBDIR,
    METADATA_FILENAME,
    candidate_bucket_counts,
    filter_metadata,
    load_metadata_raw,
)
from Sandbox.singerclassifier.utils import (
    REQUIRED_METADATA_COLUMNS,
    age_histogram_by_decade,
    age_summary_stats,
    count_m4a_files,
    ensure_parent_dir,
    print_table,
    recordings_per_account_summary,
    save_json,
    validate_required_columns,
    value_counts_dict,
)


def inspect_data(
    data_root: Path,
    summary_json: Path,
    min_age: int,
    max_age: int,
) -> dict:
    data_root = Path(data_root)
    tsv_path = data_root / METADATA_FILENAME
    audio_dir = data_root / AUDIO_SUBDIR

    raw_df = load_metadata_raw(data_root)
    validate_required_columns(raw_df, REQUIRED_METADATA_COLUMNS)

    cleaned_df, drop_counts = filter_metadata(
        raw_df,
        data_root,
        min_age=min_age,
        max_age=max_age,
    )

    audio_dir_exists = audio_dir.is_dir()
    num_m4a = count_m4a_files(audio_dir)
    usable_rows = drop_counts["usable_rows"]

    age_stats = age_summary_stats(cleaned_df["age"]) if usable_rows else {}
    age_histogram = (
        age_histogram_by_decade(cleaned_df["age"]) if usable_rows else {}
    )
    bucket_counts = candidate_bucket_counts(cleaned_df) if usable_rows else {}

    summary = {
        "data_root": str(data_root),
        "tsv_exists": tsv_path.is_file(),
        "audio_dir_exists": audio_dir_exists,
        "min_age": min_age,
        "max_age": max_age,
        "raw_metadata_rows": drop_counts["raw_metadata_rows"],
        "num_m4a_files": num_m4a,
        "usable_rows": usable_rows,
        "drop_counts": drop_counts,
        "age_summary": age_stats,
        "age_histogram_by_decade": age_histogram,
        "candidate_bucket_counts": bucket_counts,
        "unique_accounts": int(cleaned_df["account_id"].nunique())
        if usable_rows
        else 0,
        "recordings_per_account": recordings_per_account_summary(cleaned_df)
        if usable_rows
        else {},
    }

    if "gender" in cleaned_df.columns and usable_rows:
        summary["gender_counts"] = value_counts_dict(cleaned_df["gender"])

    if "country" in cleaned_df.columns and usable_rows:
        summary["country_top_20"] = value_counts_dict(
            cleaned_df["country"], top_n=20
        )

    if "device_os" in cleaned_df.columns and usable_rows:
        summary["device_os_counts"] = value_counts_dict(cleaned_df["device_os"])

    print_table("Dataset root", {"path": data_root})
    print_table("Files", {
        "tsv_exists": summary["tsv_exists"],
        "audio_dir_exists": summary["audio_dir_exists"],
        "num_m4a_files": num_m4a,
    })
    print_table("Row counts", {
        "raw_metadata_rows": drop_counts["raw_metadata_rows"],
        "usable_rows": usable_rows,
        "dropped_invalid_birth_year": drop_counts["dropped_invalid_birth_year"],
        "dropped_invalid_creation_timestamp": drop_counts[
            "dropped_invalid_creation_timestamp"
        ],
        "dropped_missing_audio": drop_counts["dropped_missing_audio"],
        "dropped_invalid_age": drop_counts["dropped_invalid_age"],
    })

    if usable_rows:
        print_table("Age summary", {
            "min": age_stats["min"],
            "max": age_stats["max"],
            "mean": f"{age_stats['mean']:.2f}",
            "median": age_stats["median"],
        })
        print_table("Age histogram (by decade)", age_histogram)

        for scheme_name, counts in bucket_counts.items():
            print_table(f"Candidate buckets: {scheme_name}", counts)

        print_table("Accounts", {
            "unique_accounts": summary["unique_accounts"],
            **summary["recordings_per_account"],
        })

        if "gender_counts" in summary:
            print_table("Gender", summary["gender_counts"])
        if "country_top_20" in summary:
            print_table("Country (top 20)", summary["country_top_20"])
        if "device_os_counts" in summary:
            print_table("Device OS", summary["device_os_counts"])

    ensure_parent_dir(summary_json)
    save_json(summary_json, summary)
    print(f"\nSaved inspection summary to: {summary_json}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DAMP-S-AG dataset")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to DAMP-S-AG folder containing amazing_grace.tsv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("experiments/data_inspection/data_summary.json"),
        help="Path to save JSON inspection summary",
    )
    parser.add_argument("--min-age", type=int, default=10)
    parser.add_argument("--max-age", type=int, default=90)
    args = parser.parse_args()

    inspect_data(
        data_root=args.data_root,
        summary_json=args.summary_json,
        min_age=args.min_age,
        max_age=args.max_age,
    )


if __name__ == "__main__":
    main()
