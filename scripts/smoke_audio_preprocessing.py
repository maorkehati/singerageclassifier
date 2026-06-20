"""Smoke test for DAMP-S-AG audio preprocessing and dataloaders."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from Sandbox.singerclassifier.data import DampSAGDataset, build_dataloader


def _print_tensor_stats(name: str, x: torch.Tensor) -> None:
    print(f"{name} shape: {tuple(x.shape)}")
    print(f"{name} min/max/mean/std: {x.min().item():.4f} / {x.max().item():.4f} / "
          f"{x.mean().item():.4f} / {x.std().item():.4f}")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def _expected_time_frames(duration_sec: float, sample_rate: int, hop_length: int) -> int:
    num_samples = int(duration_sec * sample_rate)
    return 1 + (num_samples - 1) // hop_length


def smoke_split(
    split_csv: Path,
    split: str,
    batch_size: int,
    duration_sec: float,
    sample_rate: int,
    n_mels: int,
    hop_length: int,
    num_workers: int,
    return_waveform: bool,
) -> None:
    dataset = DampSAGDataset(
        split_csv=split_csv,
        split=split,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        hop_length=hop_length,
        n_mels=n_mels,
        return_waveform=return_waveform,
    )
    print(f"\n{split} dataset length: {len(dataset)}")

    x, y = dataset[0]
    _print_tensor_stats(f"{split} sample input", x)
    print(f"{split} sample label: {y}")

    meta = DampSAGDataset(
        split_csv=split_csv,
        split=split,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        hop_length=hop_length,
        n_mels=n_mels,
        return_metadata=True,
        return_waveform=return_waveform,
    )[0]
    print(f"{split} sample metadata:")
    for key in (
        "performance_id",
        "account_id",
        "age",
        "age_bucket",
        "age_bucket_id",
        "audio_path",
        "split",
    ):
        if key in meta:
            print(f"  {key}: {meta[key]}")

    loader = build_dataloader(
        split_csv=split_csv,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        hop_length=hop_length,
        n_mels=n_mels,
        return_waveform=return_waveform,
    )

    batch_x, batch_y = next(iter(loader))
    _print_tensor_stats(f"{split} batch input", batch_x)
    print(f"{split} batch label shape: {tuple(batch_y.shape)}")
    print(f"{split} batch labels: {batch_y.tolist()}")

    if return_waveform:
        expected_samples = int(sample_rate * duration_sec)
        expected = (min(batch_size, len(dataset)), 1, expected_samples)
        if tuple(batch_x.shape) != expected:
            print(
                f"WARNING: expected waveform batch shape {expected}, "
                f"got {tuple(batch_x.shape)}"
            )
    else:
        expected_time = _expected_time_frames(duration_sec, sample_rate, hop_length)
        print(f"{split} expected time frames (approx): {expected_time}")
        if batch_x.ndim != 4 or batch_x.shape[1] != 1 or batch_x.shape[2] != n_mels:
            print(
                f"WARNING: expected batch shape [batch, 1, {n_mels}, time], "
                f"got {tuple(batch_x.shape)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test DAMP-S-AG audio preprocessing"
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        required=True,
        help="Path to damp_sag_splits.csv",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--duration-sec", type=float, default=15.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--return-waveform",
        action="store_true",
        help="Return fixed-length waveforms instead of log-mel spectrograms",
    )
    args = parser.parse_args()

    if not args.split_csv.is_file():
        raise FileNotFoundError(f"Split CSV not found: {args.split_csv}")

    for split in ("train", "val", "test"):
        smoke_split(
            split_csv=args.split_csv,
            split=split,
            batch_size=args.batch_size,
            duration_sec=args.duration_sec,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            hop_length=args.hop_length,
            num_workers=args.num_workers,
            return_waveform=args.return_waveform,
        )

    mode = "waveform" if args.return_waveform else "log-mel spectrogram"
    print(f"\nAudio preprocessing smoke test passed ({mode} mode).")


if __name__ == "__main__":
    main()
