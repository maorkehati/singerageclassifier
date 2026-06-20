"""Metadata loading, age computation, and PyTorch dataset for DAMP-S-AG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .utils import (
    REQUIRED_METADATA_COLUMNS,
    compute_age,
    normalize_nulls,
    parse_int_series,
    validate_required_columns,
)

METADATA_FILENAME = "amazing_grace.tsv"
AUDIO_SUBDIR = "amazing_grace"

DEFAULT_MIN_AGE = 10
DEFAULT_MAX_AGE = 90

DEFAULT_BUCKET_THRESHOLDS = (25, 35)
DEFAULT_BUCKET_NAMES = ("under_25", "age_25_34", "age_35_plus")

CANDIDATE_3CLASS = {
    "name": "3-class default",
    "thresholds": (25, 35),
    "bucket_names": ("under_25", "age_25_34", "age_35_plus"),
}

CANDIDATE_4CLASS = {
    "name": "4-class candidate",
    "thresholds": (25, 35, 50),
    "bucket_names": ("under_25", "age_25_34", "age_35_49", "age_50_plus"),
}


def _default_bucket_names(thresholds: tuple[int, ...]) -> tuple[str, ...]:
    if thresholds == (25, 35):
        return DEFAULT_BUCKET_NAMES
    if thresholds == (25, 35, 50):
        return CANDIDATE_4CLASS["bucket_names"]
    names = []
    for idx, threshold in enumerate(thresholds):
        if idx == 0:
            names.append(f"under_{threshold}")
        else:
            prev = thresholds[idx - 1]
            names.append(f"age_{prev}_{threshold - 1}")
    names.append(f"age_{thresholds[-1]}_plus")
    return tuple(names)


def filter_metadata(
    df: pd.DataFrame,
    data_root: Path,
    min_age: int = DEFAULT_MIN_AGE,
    max_age: int = DEFAULT_MAX_AGE,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply metadata filters and return cleaned rows plus drop counts."""
    data_root = Path(data_root)
    out = df.copy()
    raw_count = len(out)

    out["performance_id"] = out["performance_id"].astype(str).str.strip()
    out["account_id"] = out["account_id"].astype(str).str.strip()
    out["birth_year"] = parse_int_series(out["birth_year"])
    out["creation_timestamp"] = parse_int_series(out["creation_timestamp"])

    audio_dir = data_root / AUDIO_SUBDIR
    out["audio_path"] = out["performance_id"].map(
        lambda pid: str(audio_dir / f"{pid}.m4a")
    )

    valid_performance = out["performance_id"].notna() & (out["performance_id"] != "")
    valid_account = out["account_id"].notna() & (out["account_id"] != "")
    valid_birth = out["birth_year"].notna()
    valid_timestamp = out["creation_timestamp"].notna()
    file_exists = out["audio_path"].map(lambda p: Path(p).is_file())

    drop_counts = {
        "raw_metadata_rows": raw_count,
        "dropped_invalid_performance_id": int((~valid_performance).sum()),
        "dropped_invalid_account_id": int((valid_performance & ~valid_account).sum()),
        "dropped_invalid_birth_year": int(
            (valid_performance & valid_account & ~valid_birth).sum()
        ),
        "dropped_invalid_creation_timestamp": int(
            (
                valid_performance
                & valid_account
                & valid_birth
                & ~valid_timestamp
            ).sum()
        ),
        "dropped_missing_audio": int(
            (
                valid_performance
                & valid_account
                & valid_birth
                & valid_timestamp
                & ~file_exists
            ).sum()
        ),
    }

    filtered = out[
        valid_performance
        & valid_account
        & valid_birth
        & valid_timestamp
        & file_exists
    ].copy()

    filtered["age"] = compute_age(filtered["birth_year"], filtered["creation_timestamp"])
    invalid_age = ~filtered["age"].between(min_age, max_age)
    drop_counts["dropped_invalid_age"] = int(invalid_age.sum())

    filtered = filtered[~invalid_age].copy()
    filtered["age"] = filtered["age"].astype(int)
    drop_counts["usable_rows"] = int(len(filtered))

    return filtered.reset_index(drop=True), drop_counts


def assign_age_buckets(
    df: pd.DataFrame,
    thresholds: tuple[int, ...] = DEFAULT_BUCKET_THRESHOLDS,
    bucket_names: tuple[str, ...] | None = None,
    warn_on_empty: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assign age_bucket and age_bucket_id columns from age values."""
    if len(thresholds) < 1:
        raise ValueError("thresholds must contain at least one value")

    if bucket_names is None:
        bucket_names = _default_bucket_names(thresholds)

    expected_names = len(thresholds) + 1
    if len(bucket_names) != expected_names:
        raise ValueError(
            f"Expected {expected_names} bucket names for {len(thresholds)} thresholds"
        )

    out = df.copy()
    age = out["age"]
    bucket = pd.Series(pd.NA, index=out.index, dtype="object")

    bucket = bucket.mask(age < thresholds[0], bucket_names[0])
    for idx in range(1, len(thresholds)):
        low = thresholds[idx - 1]
        high = thresholds[idx]
        bucket = bucket.mask((age >= low) & (age < high), bucket_names[idx])
    bucket = bucket.mask(age >= thresholds[-1], bucket_names[-1])
    out["age_bucket"] = bucket

    mapping = {name: idx for idx, name in enumerate(bucket_names)}
    out["age_bucket_id"] = out["age_bucket"].map(mapping).astype("Int64")

    for name in bucket_names:
        count = int((out["age_bucket"] == name).sum())
        if warn_on_empty and count == 0:
            print(f"WARNING: age bucket '{name}' is empty after assignment.")

    return out, mapping


def candidate_bucket_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return counts for predefined 3-class and 4-class bucket schemes."""
    results: dict[str, dict[str, int]] = {}
    for scheme in (CANDIDATE_3CLASS, CANDIDATE_4CLASS):
        bucketed, _ = assign_age_buckets(
            df,
            thresholds=scheme["thresholds"],
            bucket_names=scheme["bucket_names"],
            warn_on_empty=False,
        )
        results[scheme["name"]] = {
            name: int((bucketed["age_bucket"] == name).sum())
            for name in scheme["bucket_names"]
        }
    return results


def load_metadata_raw(data_root: Path) -> pd.DataFrame:
    """Load the TSV metadata without filtering."""
    tsv_path = data_root / METADATA_FILENAME
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    return normalize_nulls(df)


def load_metadata(
    data_root: Path,
    min_age: int = DEFAULT_MIN_AGE,
    max_age: int = DEFAULT_MAX_AGE,
) -> pd.DataFrame:
    """Load, validate, filter, and enrich DAMP-S-AG metadata."""
    data_root = Path(data_root)
    df = load_metadata_raw(data_root)
    validate_required_columns(df, REQUIRED_METADATA_COLUMNS)
    filtered, _ = filter_metadata(df, data_root, min_age=min_age, max_age=max_age)
    return filtered


def _crop_or_pad(
    waveform: torch.Tensor,
    target_samples: int,
    random_crop: bool,
) -> torch.Tensor:
    """Crop or zero-pad waveform along the last dimension."""
    num_samples = waveform.shape[-1]
    if num_samples > target_samples:
        if random_crop:
            start = torch.randint(0, num_samples - target_samples + 1, (1,)).item()
        else:
            start = (num_samples - target_samples) // 2
        return waveform[..., start : start + target_samples]
    if num_samples < target_samples:
        pad = target_samples - num_samples
        return torch.nn.functional.pad(waveform, (0, pad))
    return waveform


class DampSAGDataset(Dataset):
    """PyTorch dataset for DAMP-S-AG age-bucket classification."""

    def __init__(
        self,
        split_csv: str | Path,
        split: str,
        sample_rate: int = 22050,
        duration_sec: float = 15.0,
        random_crop: bool = False,
        mono: bool = True,
        return_metadata: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.split = split
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.random_crop = random_crop
        self.mono = mono
        self.return_metadata = return_metadata
        self.target_samples = int(sample_rate * duration_sec)

        if not self.split_csv.is_file():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")

        df = pd.read_csv(self.split_csv)
        if "split" not in df.columns:
            raise ValueError(f"Split CSV missing 'split' column: {self.split_csv}")

        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(
                f"No rows available for split '{split}' in {self.split_csv}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Any:
        row = self.df.iloc[index]
        audio_path = Path(str(row["audio_path"]))

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found for index {index}: {audio_path}"
            )

        try:
            waveform, sr = torchaudio.load(str(audio_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio for index {index}: {audio_path}"
            ) from exc

        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        waveform = _crop_or_pad(
            waveform, self.target_samples, self.random_crop
        )
        waveform = waveform.to(torch.float32)

        label = int(row["age_bucket_id"])

        if self.return_metadata:
            return {
                "waveform": waveform,
                "label": label,
                "performance_id": row["performance_id"],
                "account_id": row["account_id"],
                "age": int(row["age"]),
                "age_bucket": row["age_bucket"],
                "audio_path": str(audio_path),
            }

        return waveform, label


def build_dataloader(
    split_csv: str | Path,
    split: str,
    batch_size: int = 16,
    num_workers: int = 2,
    sample_rate: int = 22050,
    duration_sec: float = 15.0,
    random_crop: bool | None = None,
    mono: bool = True,
    return_metadata: bool = False,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build a DataLoader for the requested split."""
    if random_crop is None:
        random_crop = split == "train"
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset = DampSAGDataset(
        split_csv=split_csv,
        split=split,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        random_crop=random_crop,
        mono=mono,
        return_metadata=return_metadata,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
