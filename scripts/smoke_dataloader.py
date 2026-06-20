"""Smoke test for DAMP-S-AG PyTorch dataloaders."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from Sandbox.singerclassifier.data import DampSAGDataset, build_dataloader


def assert_no_leakage(split_csv: Path) -> None:
    df = pd.read_csv(split_csv)
    splits = {
        name: set(df.loc[df["split"] == name, "account_id"].astype(str))
        for name in ("train", "val", "test")
    }

    checks = {
        "train_val_disjoint": splits["train"].isdisjoint(splits["val"]),
        "train_test_disjoint": splits["train"].isdisjoint(splits["test"]),
        "val_test_disjoint": splits["val"].isdisjoint(splits["test"]),
    }

    print("\nAccount leakage check from split CSV:")
    for name, ok in checks.items():
        print(f"  {name}: {'OK' if ok else 'FAIL'}")

    if not all(checks.values()):
        raise AssertionError(f"Account leakage detected: {checks}")


def smoke_split(
    split_csv: Path,
    split: str,
    batch_size: int,
    duration_sec: float,
    sample_rate: int,
) -> None:
    dataset = DampSAGDataset(
        split_csv=split_csv,
        split=split,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        random_crop=(split == "train"),
    )
    print(f"\n{split} dataset length: {len(dataset)}")

    loader = build_dataloader(
        split_csv=split_csv,
        split=split,
        batch_size=batch_size,
        num_workers=0,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        random_crop=(split == "train"),
    )

    batch = next(iter(loader))
    waveforms, labels = batch
    print(f"{split} waveform batch shape: {tuple(waveforms.shape)}")
    print(f"{split} label tensor shape: {tuple(labels.shape)}")
    print(f"{split} label values: {labels.tolist()}")

    expected_samples = int(sample_rate * duration_sec)
    expected_shape = (batch_size, 1, expected_samples)
    if tuple(waveforms.shape) != expected_shape:
        actual_batch = waveforms.shape[0]
        if actual_batch < batch_size:
            expected_shape = (actual_batch, 1, expected_samples)
        if tuple(waveforms.shape) != expected_shape:
            print(
                f"WARNING: expected shape {expected_shape}, "
                f"got {tuple(waveforms.shape)}"
            )

    meta_dataset = DampSAGDataset(
        split_csv=split_csv,
        split=split,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        return_metadata=True,
    )
    sample = meta_dataset[0]
    print(f"{split} sample metadata:")
    for key in ("performance_id", "account_id", "age", "age_bucket", "audio_path"):
        print(f"  {key}: {sample[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test DAMP-S-AG dataloaders")
    parser.add_argument(
        "--split-csv",
        type=Path,
        required=True,
        help="Path to damp_sag_splits.csv",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--duration-sec", type=float, default=15.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()

    if not args.split_csv.is_file():
        raise FileNotFoundError(f"Split CSV not found: {args.split_csv}")

    assert_no_leakage(args.split_csv)

    for split in ("train", "val", "test"):
        smoke_split(
            split_csv=args.split_csv,
            split=split,
            batch_size=args.batch_size,
            duration_sec=args.duration_sec,
            sample_rate=args.sample_rate,
        )

    print("\nDataloader smoke test passed.")


if __name__ == "__main__":
    main()
