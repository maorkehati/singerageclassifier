"""Leakage-safe train/val/test split creation for DAMP-S-AG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .utils import (
    SPLIT_CSV_OPTIONAL_COLUMNS,
    SPLIT_CSV_REQUIRED_COLUMNS,
    SPLIT_CSV_PATH,
    ensure_parent_dir,
    percentage_dict,
    save_json,
    validate_required_columns,
    value_counts_dict,
)

ALGORITHM_NAME = "account_level_stratified_shuffle_split_by_modal_age_bucket"
SPLIT_NAMES = ("train", "val", "test")


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    tolerance: float = 1e-6,
) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=tolerance):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )


def validate_labeled_metadata(df: pd.DataFrame) -> None:
    """Validate cleaned, labeled metadata before split creation."""
    if df.empty:
        raise ValueError("Cleaned metadata dataframe is empty after filtering.")

    required = (
        "performance_id",
        "account_id",
        "audio_path",
        "birth_year",
        "creation_timestamp",
        "age",
        "age_bucket",
        "age_bucket_id",
    )
    validate_required_columns(df, required)

    missing_audio = df[
        ~df["audio_path"].astype(str).map(lambda p: Path(p).is_file())
    ]
    if not missing_audio.empty:
        sample = missing_audio["audio_path"].iloc[0]
        raise ValueError(
            f"{len(missing_audio)} rows have missing audio files. Example: {sample}"
        )


def _modal_age_bucket_id(account_df: pd.DataFrame) -> int:
    counts = account_df["age_bucket_id"].value_counts()
    return int(counts.index[0])


def build_account_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per account with modal age_bucket_id for stratification."""
    rows = []
    for account_id, group in df.groupby("account_id"):
        rows.append(
            {
                "account_id": account_id,
                "strat_label": _modal_age_bucket_id(group),
                "num_recordings": len(group),
            }
        )
    return pd.DataFrame(rows)


def _stratified_account_split(
    account_table: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str], bool]:
    """Split account IDs with stratification; fall back to random shuffle."""
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    accounts = account_table["account_id"].astype(str).tolist()
    labels = account_table["strat_label"].astype(int).tolist()
    n_accounts = len(accounts)

    val_test_ratio = val_ratio + test_ratio
    test_fraction_of_temp = test_ratio / val_test_ratio
    fallback_used = False

    try:
        sss_train = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_ratio,
            random_state=seed,
        )
        train_idx, temp_idx = next(sss_train.split(np.zeros(n_accounts), labels))

        temp_accounts = [accounts[i] for i in temp_idx]
        temp_labels = [labels[i] for i in temp_idx]

        if len(temp_accounts) < 2:
            raise ValueError("Not enough accounts in temp split for stratification.")

        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            train_size=1.0 - test_fraction_of_temp,
            random_state=seed + 1,
        )
        val_local, test_local = next(
            sss_val.split(np.zeros(len(temp_accounts)), temp_labels)
        )

        train_accounts = {accounts[i] for i in train_idx}
        val_accounts = {temp_accounts[i] for i in val_local}
        test_accounts = {temp_accounts[i] for i in test_local}

    except Exception as exc:
        fallback_used = True
        print(
            "WARNING: Stratified account split failed "
            f"({exc}). Falling back to shuffled account split."
        )
        shuffled = account_table.sample(frac=1, random_state=seed)[
            "account_id"
        ].astype(str).tolist()

        n_train = int(round(n_accounts * train_ratio))
        n_val = int(round(n_accounts * val_ratio))
        n_test = n_accounts - n_train - n_val
        if n_test <= 0:
            n_test = max(1, n_accounts - n_train - n_val)
            n_val = n_accounts - n_train - n_test

        train_accounts = set(shuffled[:n_train])
        val_accounts = set(shuffled[n_train : n_train + n_val])
        test_accounts = set(shuffled[n_train + n_val :])

    return train_accounts, val_accounts, test_accounts, fallback_used


def compute_leakage_check(
    train_accounts: set[str],
    val_accounts: set[str],
    test_accounts: set[str],
) -> dict[str, int | bool]:
    train_val_overlap = len(train_accounts & val_accounts)
    train_test_overlap = len(train_accounts & test_accounts)
    val_test_overlap = len(val_accounts & test_accounts)
    passed = (
        train_val_overlap == 0
        and train_test_overlap == 0
        and val_test_overlap == 0
    )
    return {
        "passed": passed,
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
    }


def assert_no_account_leakage(
    train_accounts: set[str],
    val_accounts: set[str],
    test_accounts: set[str],
) -> dict[str, int | bool]:
    leakage_check = compute_leakage_check(
        train_accounts, val_accounts, test_accounts
    )
    if not leakage_check["passed"]:
        raise AssertionError(f"Account leakage detected: {leakage_check}")
    return leakage_check


def assign_splits(
    df: pd.DataFrame,
    train_accounts: set[str],
    val_accounts: set[str],
    test_accounts: set[str],
) -> pd.DataFrame:
    out = df.copy()
    account_ids = out["account_id"].astype(str)
    split = pd.Series(pd.NA, index=out.index, dtype="object")
    split = split.mask(account_ids.isin(train_accounts), "train")
    split = split.mask(account_ids.isin(val_accounts), "val")
    split = split.mask(account_ids.isin(test_accounts), "test")

    unassigned = split.isna()
    if unassigned.any():
        raise ValueError(
            f"{int(unassigned.sum())} rows could not be assigned to a split."
        )

    out["split"] = split
    return out


def validate_split_coverage(
    split_df: pd.DataFrame,
    bucket_names: tuple[str, ...],
) -> None:
    """Ensure all splits exist and warn about missing age buckets."""
    for split_name in SPLIT_NAMES:
        split_rows = split_df[split_df["split"] == split_name]
        if split_rows.empty:
            raise ValueError(f"Split '{split_name}' is empty.")

    for split_name in SPLIT_NAMES:
        split_rows = split_df[split_df["split"] == split_name]
        present = set(split_rows["age_bucket"].astype(str))
        for bucket_name in bucket_names:
            if bucket_name not in present:
                print(
                    f"WARNING: age bucket '{bucket_name}' is missing from "
                    f"split '{split_name}'."
                )


def select_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select required and available optional columns for the split CSV."""
    columns = list(SPLIT_CSV_REQUIRED_COLUMNS)
    for col in SPLIT_CSV_OPTIONAL_COLUMNS:
        if col in df.columns:
            columns.append(col)
    return df[columns].copy()


def _split_summary_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.crosstab(df["split"], df[column])


def _build_summary(
    split_df: pd.DataFrame,
    class_mapping: dict[str, int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    fallback_used: bool,
    leakage_check: dict[str, int | bool],
    output_csv: Path,
) -> dict[str, Any]:
    age_bucket_counts = {
        split_name: value_counts_dict(
            split_df.loc[split_df["split"] == split_name, "age_bucket"]
        )
        for split_name in SPLIT_NAMES
    }

    summary: dict[str, Any] = {
        "total_rows": int(len(split_df)),
        "total_accounts": int(split_df["account_id"].nunique()),
        "class_mapping": class_mapping,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "seed": seed,
        "algorithm": ALGORITHM_NAME,
        "fallback_used": fallback_used,
        "rows_per_split": value_counts_dict(split_df["split"]),
        "accounts_per_split": {
            split_name: int(
                split_df.loc[split_df["split"] == split_name, "account_id"].nunique()
            )
            for split_name in SPLIT_NAMES
        },
        "age_bucket_counts_per_split": age_bucket_counts,
        "age_bucket_percentages_per_split": {
            split_name: percentage_dict(counts)
            for split_name, counts in age_bucket_counts.items()
        },
        "leakage_check": leakage_check,
        "output_csv": str(output_csv),
    }

    if "gender" in split_df.columns:
        summary["gender_counts_per_split"] = {
            split_name: value_counts_dict(
                split_df.loc[split_df["split"] == split_name, "gender"]
            )
            for split_name in SPLIT_NAMES
        }

    if "country" in split_df.columns:
        summary["country_top_counts_per_split"] = {
            split_name: value_counts_dict(
                split_df.loc[split_df["split"] == split_name, "country"],
                top_n=20,
            )
            for split_name in SPLIT_NAMES
        }

    if "device_os" in split_df.columns:
        summary["device_os_counts_per_split"] = {
            split_name: value_counts_dict(
                split_df.loc[split_df["split"] == split_name, "device_os"]
            )
            for split_name in SPLIT_NAMES
        }

    return summary


def print_split_summary(
    summary: dict[str, Any],
    output_csv: Path,
    summary_json: Path,
) -> None:
    print("\nSplit creation complete")
    print("-----------------------")
    print(f"Algorithm: {summary['algorithm']}")
    print(f"Fallback used: {summary['fallback_used']}")
    print(f"Output CSV: {output_csv}")
    print(f"Summary JSON: {summary_json}")

    print("\nRows per split")
    print("--------------")
    for split_name in SPLIT_NAMES:
        print(f"{split_name}: {summary['rows_per_split'].get(split_name, 0)}")

    print("\nAccounts per split")
    print("------------------")
    for split_name in SPLIT_NAMES:
        print(f"{split_name}: {summary['accounts_per_split'].get(split_name, 0)}")

    print("\nAge buckets per split")
    print("---------------------")
    bucket_names = list(summary["class_mapping"].keys())
    header = "split\t" + "\t".join(bucket_names)
    print(header)
    for split_name in SPLIT_NAMES:
        counts = summary["age_bucket_counts_per_split"].get(split_name, {})
        row = [split_name] + [str(counts.get(name, 0)) for name in bucket_names]
        print("\t".join(row))

    leakage = summary["leakage_check"]
    print("\nLeakage check")
    print("-------------")
    print(f"train/val overlap: {leakage['train_val_overlap']}")
    print(f"train/test overlap: {leakage['train_test_overlap']}")
    print(f"val/test overlap: {leakage['val_test_overlap']}")
    print(f"passed: {leakage['passed']}")

    for field, title in (
        ("gender_counts_per_split", "Gender counts per split"),
        ("device_os_counts_per_split", "Device OS counts per split"),
        ("country_top_counts_per_split", "Country top counts per split"),
    ):
        if field not in summary:
            continue
        print(f"\n{title}")
        print("-" * len(title))
        for split_name in SPLIT_NAMES:
            counts = summary[field].get(split_name, {})
            compact = ", ".join(f"{k}={v}" for k, v in list(counts.items())[:5])
            if len(counts) > 5:
                compact += ", ..."
            print(f"  {split_name}: {compact}")


def create_splits(
    df: pd.DataFrame,
    class_mapping: dict[str, int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_csv: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create leakage-safe account-level splits."""
    validate_labeled_metadata(df)
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    account_table = build_account_table(df)
    train_accounts, val_accounts, test_accounts, fallback_used = (
        _stratified_account_split(
            account_table,
            train_ratio,
            val_ratio,
            test_ratio,
            seed,
        )
    )

    leakage_check = assert_no_account_leakage(
        train_accounts, val_accounts, test_accounts
    )

    split_df = assign_splits(df, train_accounts, val_accounts, test_accounts)
    bucket_names = tuple(class_mapping.keys())
    validate_split_coverage(split_df, bucket_names)

    summary = _build_summary(
        split_df=split_df,
        class_mapping=class_mapping,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        fallback_used=fallback_used,
        leakage_check=leakage_check,
        output_csv=output_csv or SPLIT_CSV_PATH,
    )

    return split_df, summary


def save_splits(
    df: pd.DataFrame,
    output_csv: Path,
    summary: dict[str, Any],
    summary_json: Path,
) -> None:
    ensure_parent_dir(output_csv)
    output_df = select_split_columns(df)
    output_df.to_csv(output_csv, index=False)
    save_json(summary_json, summary)
