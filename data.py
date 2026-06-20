"""Metadata loading, age computation, and PyTorch dataset for DAMP-S-AG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import (
    audio_cache_path,
    crop_or_pad_waveform,
    load_audio,
    load_cached_waveform,
)
from .features import LogMelSpectrogram, SpectrogramAugment
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


DATASET_REQUIRED_COLUMNS = (
    "audio_path",
    "age_bucket_id",
    "performance_id",
    "account_id",
    "split",
)


class DampSAGDataset(Dataset):
    """PyTorch dataset for DAMP-S-AG age-bucket classification."""

    def __init__(
        self,
        split_csv: str | Path,
        split: str,
        sample_rate: int = 22050,
        duration_sec: float = 15.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 80,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        random_crop: bool | None = None,
        augment_train: bool = False,
        augmentation_cfg: dict | None = None,
        return_metadata: bool = False,
        return_waveform: bool = False,
        use_audio_cache: bool = False,
        audio_cache_dir: str | Path | None = None,
        strict_audio_cache: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.split = split
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.random_crop = random_crop if random_crop is not None else split == "train"
        self.return_metadata = return_metadata
        self.return_waveform = return_waveform
        self.use_audio_cache = use_audio_cache
        self.audio_cache_dir = Path(audio_cache_dir) if audio_cache_dir is not None else None
        self.strict_audio_cache = strict_audio_cache
        self.target_samples = int(sample_rate * duration_sec)
        self.augment_train = augment_train and split == "train"
        augmentation_cfg = augmentation_cfg or {}
        self.waveform_noise_std = float(
            augmentation_cfg.get("waveform_noise_std", 0.0)
        ) if self.augment_train else 0.0

        if not self.split_csv.is_file():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")

        df = pd.read_csv(self.split_csv)
        validate_required_columns(df, DATASET_REQUIRED_COLUMNS)

        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(
                f"No rows available for split '{split}' in {self.split_csv}"
            )

        self.feature_extractor = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            normalize=True,
        )
        self.spec_augment = None
        if self.augment_train and augmentation_cfg.get("enabled", False):
            self.spec_augment = SpectrogramAugment(
                time_mask_cfg=augmentation_cfg.get("time_mask"),
                freq_mask_cfg=augmentation_cfg.get("freq_mask"),
            )

        if self.use_audio_cache and self.audio_cache_dir is None:
            raise ValueError("audio_cache_dir is required when use_audio_cache=True")

    def _load_waveform(self, row: pd.Series, index: int, audio_path: Path) -> torch.Tensor:
        performance_id = str(row["performance_id"])

        if self.use_audio_cache:
            cache_path = audio_cache_path(
                self.audio_cache_dir,
                performance_id,
                source_path=audio_path,
            )
            if cache_path.is_file():
                try:
                    waveform, _ = load_cached_waveform(
                        cache_path,
                        expected_sample_rate=self.sample_rate,
                    )
                    return waveform
                except Exception as exc:
                    if self.strict_audio_cache:
                        raise RuntimeError(
                            f"Invalid audio cache for index {index}: {cache_path}"
                        ) from exc
                    print(
                        f"WARNING: invalid cache {cache_path}, falling back to decode: {exc}"
                    )
            elif self.strict_audio_cache:
                raise FileNotFoundError(
                    f"Audio cache missing for performance_id={performance_id}: {cache_path}\n"
                    "Precompute the cache with:\n"
                    "  python -m Sandbox.singerclassifier.scripts.precompute_audio_cache "
                    f"--split-csv {self.split_csv} --cache-dir {self.audio_cache_dir}"
                )

        try:
            waveform, _ = load_audio(
                audio_path,
                target_sample_rate=self.sample_rate,
                mono=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio for index {index}: {audio_path}"
            ) from exc
        return waveform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Any:
        row = self.df.iloc[index]
        audio_path = Path(str(row["audio_path"]))

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found for index {index}: {audio_path}"
            )

        waveform = self._load_waveform(row, index, audio_path)

        if self.augment_train and self.waveform_noise_std > 0.0:
            waveform = waveform + torch.randn_like(waveform) * self.waveform_noise_std

        waveform = crop_or_pad_waveform(
            waveform,
            target_num_samples=self.target_samples,
            random_crop=self.random_crop,
        )

        label = int(row["age_bucket_id"])

        if self.return_waveform:
            x = waveform.to(torch.float32)
        else:
            x = self.feature_extractor(waveform).to(torch.float32)
            if self.spec_augment is not None:
                x = self.spec_augment(x)

        if self.return_metadata:
            metadata = {
                "input": x,
                "label": label,
                "performance_id": row["performance_id"],
                "account_id": row["account_id"],
                "audio_path": str(audio_path),
                "split": row["split"],
                "age_bucket_id": label,
            }
            if "age" in row and pd.notna(row["age"]):
                metadata["age"] = int(row["age"])
            if "age_bucket" in row and pd.notna(row["age_bucket"]):
                metadata["age_bucket"] = row["age_bucket"]
            return metadata

        return x, label


def build_dataloader(
    split_csv: str | Path,
    split: str,
    batch_size: int = 16,
    num_workers: int = 2,
    sample_rate: int = 22050,
    duration_sec: float = 15.0,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 80,
    f_min: float = 50.0,
    f_max: float = 8000.0,
    random_crop: bool | None = None,
    augment_train: bool = False,
    augmentation_cfg: dict | None = None,
    return_metadata: bool = False,
    return_waveform: bool = False,
    use_audio_cache: bool = False,
    audio_cache_dir: str | Path | None = None,
    strict_audio_cache: bool = False,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build a DataLoader for the requested split."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset = DampSAGDataset(
        split_csv=split_csv,
        split=split,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        random_crop=random_crop,
        augment_train=augment_train,
        augmentation_cfg=augmentation_cfg,
        return_metadata=return_metadata,
        return_waveform=return_waveform,
        use_audio_cache=use_audio_cache,
        audio_cache_dir=audio_cache_dir,
        strict_audio_cache=strict_audio_cache,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
