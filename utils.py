"""Shared utilities for data loading and split preparation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

NULL_STRINGS = {"NULL", "null", "None", "nan", "NaN", ""}

REQUIRED_METADATA_COLUMNS = (
    "performance_id",
    "account_id",
    "birth_year",
    "creation_timestamp",
)


def normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NULL-like strings with pandas NA."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].replace(list(NULL_STRINGS), pd.NA)
    return out


def validate_required_columns(df: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_int_series(series: pd.Series) -> pd.Series:
    """Parse a series to nullable integer dtype."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def compute_age(
    birth_year: pd.Series,
    creation_timestamp: pd.Series,
) -> pd.Series:
    """Compute approximate age from birth year and Unix creation timestamp (UTC)."""
    birth = birth_year.astype("float")
    ts = creation_timestamp.astype("float")

    def _year_from_ts(value: float) -> float:
        if pd.isna(value):
            return float("nan")
        return float(datetime.fromtimestamp(value, tz=timezone.utc).year)

    recording_year = ts.map(_year_from_ts)
    return (recording_year - birth).astype("float")


def count_m4a_files(audio_dir: Path) -> int:
    if not audio_dir.is_dir():
        return 0
    return sum(1 for _ in audio_dir.glob("*.m4a"))


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def print_table(title: str, rows: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in rows.items():
        print(f"  {key}: {value}")


def value_counts_dict(series: pd.Series, top_n: int | None = None) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)
    return {str(k): int(v) for k, v in counts.items()}


def recordings_per_account_summary(
    df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.25, 0.75, 0.90, 0.95),
) -> dict[str, float | int]:
    counts = df.groupby("account_id").size()
    summary: dict[str, float | int] = {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(counts.median()),
    }
    for q in quantiles:
        summary[f"q{int(q * 100)}"] = float(counts.quantile(q))
    return summary


def age_histogram_by_decade(ages: pd.Series) -> dict[str, int]:
    """Bin ages into decade ranges such as '20-29', '30-39'."""
    valid = ages.dropna().astype(int)
    if valid.empty:
        return {}

    decade_starts = (valid // 10) * 10
    counts = decade_starts.value_counts().sort_index()
    return {f"{int(start)}-{int(start + 9)}": int(count) for start, count in counts.items()}


def age_summary_stats(ages: pd.Series) -> dict[str, float | int | None]:
    valid = ages.dropna()
    if valid.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "min": int(valid.min()),
        "max": int(valid.max()),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Int64Dtype,)):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
