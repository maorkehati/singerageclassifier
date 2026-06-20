"""Leakage-safe train/val/test split creation for DAMP-S-AG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .utils import (
    ensure_parent_dir,
    print_table,
    save_json,
    value_counts_dict,
)


def _modal_age_bucket(account_df: pd.DataFrame) -> str:
    counts = account_df["age_bucket"].value_counts()
    return str(counts.index[0])


def build_account_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per account with modal age bucket for stratification."""
    rows = []
    for account_id, group in df.groupby("account_id"):
        rows.append(
            {
                "account_id": account_id,
                "strat_label": _modal_age_bucket(group),
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
) -> tuple[set[str], set[str], set[str], str]:
    """Split account IDs with stratification; fall back to random shuffle."""
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    accounts = account_table["account_id"].tolist()
    labels = account_table["strat_label"].tolist()
    n_accounts = len(accounts)

    val_test_ratio = val_ratio + test_ratio
    test_fraction_of_temp = test_ratio / val_test_ratio

    method = "stratified"
    try:
        sss_train, _ = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_ratio,
            random_state=seed,
        ).split(np.zeros(n_accounts), labels)

        train_idx = set(sss_train[0][0])
        temp_idx = set(sss_train[0][1])

        temp_accounts = [accounts[i] for i in sorted(temp_idx)]
        temp_labels = [labels[i] for i in sorted(temp_idx)]

        if len(temp_accounts) < 2:
            raise ValueError("Not enough accounts in temp split for stratification.")

        sss_val, _ = StratifiedShuffleSplit(
            n_splits=1,
            train_size=1.0 - test_fraction_of_temp,
            random_state=seed + 1,
        ).split(np.zeros(len(temp_accounts)), temp_labels)

        val_local = set(sss_val[0][0])
        test_local = set(sss_val[0][1])

        val_accounts = {temp_accounts[i] for i in val_local}
        test_accounts = {temp_accounts[i] for i in test_local}
        train_accounts = {accounts[i] for i in train_idx}

    except Exception as exc:
        method = "random_shuffle_fallback"
        print(
            "WARNING: Stratified account split failed "
            f"({exc}). Falling back to shuffled account split."
        )
        shuffled = account_table.sample(frac=1, random_state=seed)[
            "account_id"
        ].tolist()

        n_train = int(round(n_accounts * train_ratio))
        n_val = int(round(n_accounts * val_ratio))
        n_test = n_accounts - n_train - n_val
        if n_test <= 0:
            n_test = max(1, n_accounts - n_train - n_val)
            n_val = n_accounts - n_train - n_test

        train_accounts = set(shuffled[:n_train])
        val_accounts = set(shuffled[n_train : n_train + n_val])
        test_accounts = set(shuffled[n_train + n_val :])

    return train_accounts, val_accounts, test_accounts, method


def assert_no_account_leakage(
    train_accounts: set[str],
    val_accounts: set[str],
    test_accounts: set[str],
) -> dict[str, bool]:
    checks = {
        "train_val_disjoint": train_accounts.isdisjoint(val_accounts),
        "train_test_disjoint": train_accounts.isdisjoint(test_accounts),
        "val_test_disjoint": val_accounts.isdisjoint(test_accounts),
    }
    if not all(checks.values()):
        raise AssertionError(f"Account leakage detected: {checks}")
    return checks


def assign_splits(
    df: pd.DataFrame,
    train_accounts: set[str],
    val_accounts: set[str],
    test_accounts: set[str],
) -> pd.DataFrame:
    out = df.copy()
    split = pd.Series(pd.NA, index=out.index, dtype="object")
    split = split.mask(out["account_id"].isin(train_accounts), "train")
    split = split.mask(out["account_id"].isin(val_accounts), "val")
    split = split.mask(out["account_id"].isin(test_accounts), "test")

    unassigned = split.isna()
    if unassigned.any():
        raise ValueError(
            f"{int(unassigned.sum())} rows could not be assigned to a split."
        )

    out["split"] = split
    return out


def _split_summary_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.crosstab(df["split"], df[column])


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create leakage-safe account-level splits."""
    account_table = build_account_table(df)
    train_accounts, val_accounts, test_accounts, method = _stratified_account_split(
        account_table,
        train_ratio,
        val_ratio,
        test_ratio,
        seed,
    )

    leakage_checks = assert_no_account_leakage(
        train_accounts, val_accounts, test_accounts
    )

    split_df = assign_splits(df, train_accounts, val_accounts, test_accounts)

    class_table = _split_summary_frame(split_df, "age_bucket")
    account_counts = split_df.groupby("split")["account_id"].nunique()

    print("\nAge-bucket counts by split:")
    print(class_table.to_string())

    print("\nAccount counts by split:")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {int(account_counts.get(split_name, 0))}")

    for col in ("gender", "country", "device_os"):
        if col in split_df.columns:
            print(f"\n{col} distribution by split:")
            print(_split_summary_frame(split_df, col).to_string())

    summary: dict[str, Any] = {
        "total_rows": int(len(split_df)),
        "unique_accounts": int(split_df["account_id"].nunique()),
        "split_method": method,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "seed": seed,
        "rows_per_split": value_counts_dict(split_df["split"]),
        "accounts_per_split": {
            split_name: int(split_df[split_df["split"] == split_name]["account_id"].nunique())
            for split_name in ("train", "val", "test")
        },
        "age_bucket_counts_per_split": {
            split_name: value_counts_dict(
                split_df.loc[split_df["split"] == split_name, "age_bucket"]
            )
            for split_name in ("train", "val", "test")
        },
        "leakage_checks": leakage_checks,
    }

    for col in ("gender", "country", "device_os"):
        if col in split_df.columns:
            summary[f"{col}_by_split"] = {
                split_name: value_counts_dict(
                    split_df.loc[split_df["split"] == split_name, col]
                )
                for split_name in ("train", "val", "test")
            }

    print_table("Split method", {"method": method})
    print_table("Leakage checks", leakage_checks)

    return split_df, summary


def save_splits(
    df: pd.DataFrame,
    output_csv: Path,
    summary: dict[str, Any],
    summary_json: Path,
    class_mapping: dict[str, int],
) -> None:
    ensure_parent_dir(output_csv)
    df.to_csv(output_csv, index=False)

    summary["class_mapping"] = class_mapping
    save_json(summary_json, summary)

    print(f"\nSaved split CSV to: {output_csv}")
    print(f"Saved split summary to: {summary_json}")
