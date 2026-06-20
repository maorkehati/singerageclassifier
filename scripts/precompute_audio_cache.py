"""Precompute persistent decoded waveform cache for DAMP-S-AG."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from Sandbox.singerclassifier.audio import (
    audio_cache_path,
    load_audio,
    load_cached_waveform,
    save_waveform_cache,
)
from Sandbox.singerclassifier.utils import AUDIO_CACHE_DIR, SPLIT_CSV_PATH


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _cache_is_valid(cache_path: Path, sample_rate: int) -> bool:
    try:
        load_cached_waveform(cache_path, expected_sample_rate=sample_rate)
        return True
    except Exception:
        return False


def _process_row(
    row: pd.Series,
    cache_dir: Path,
    sample_rate: int,
    overwrite: bool,
) -> dict[str, Any]:
    performance_id = str(row["performance_id"])
    source_path = Path(str(row["audio_path"]))
    cache_path = audio_cache_path(cache_dir, performance_id, source_path=source_path)

    record: dict[str, Any] = {
        "performance_id": performance_id,
        "source_path": str(source_path),
        "cache_path": str(cache_path),
        "status": "pending",
        "num_samples": "",
        "sample_rate": sample_rate,
        "source_size": "",
        "source_mtime": "",
        "error": "",
    }

    if not source_path.is_file():
        record["status"] = "failed"
        record["error"] = f"Source audio not found: {source_path}"
        return record

    stat = source_path.stat()
    record["source_size"] = int(stat.st_size)
    record["source_mtime"] = float(stat.st_mtime)

    if cache_path.is_file() and not overwrite:
        if _cache_is_valid(cache_path, sample_rate):
            waveform, _ = load_cached_waveform(cache_path, expected_sample_rate=sample_rate)
            record["status"] = "skipped_existing"
            record["num_samples"] = int(waveform.shape[-1])
            return record

    try:
        waveform, sr = load_audio(source_path, target_sample_rate=sample_rate, mono=True)
        save_waveform_cache(cache_path, waveform, sr, source_path)
        record["status"] = "cached"
        record["num_samples"] = int(waveform.shape[-1])
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)

    return record


def write_manifest(
    cache_dir: Path,
    records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "cache_manifest.csv"
    json_path = cache_dir / "cache_manifest.json"

    fieldnames = [
        "performance_id",
        "source_path",
        "cache_path",
        "status",
        "num_samples",
        "sample_rate",
        "source_size",
        "source_mtime",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "cache_dir": str(cache_dir),
        "total": len(records),
        "cached": sum(1 for r in records if r["status"] == "cached"),
        "skipped_existing": sum(1 for r in records if r["status"] == "skipped_existing"),
        "failed": sum(1 for r in records if r["status"] == "failed"),
        "records": records,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


def precompute_audio_cache(
    split_csv: Path,
    cache_dir: Path,
    sample_rate: int = 22050,
    overwrite: bool = False,
    limit: int | None = None,
    fail_fast: bool = True,
) -> dict[str, Any]:
    df = pd.read_csv(split_csv)
    if "performance_id" not in df.columns or "audio_path" not in df.columns:
        raise ValueError("split CSV must contain performance_id and audio_path columns")

    unique_df = df.drop_duplicates(subset=["performance_id"]).reset_index(drop=True)
    if limit is not None:
        unique_df = unique_df.head(limit)

    records: list[dict[str, Any]] = []
    total = len(unique_df)

    for idx, row in unique_df.iterrows():
        record = _process_row(row, cache_dir, sample_rate, overwrite)
        records.append(record)

        processed = idx + 1
        if processed % 25 == 0 or processed == total:
            print(f"Processed {processed}/{total} files...")

        if record["status"] == "failed" and fail_fast:
            write_manifest(cache_dir, records)
            raise RuntimeError(
                f"Cache precompute failed for performance_id={record['performance_id']}: "
                f"{record['error']}"
            )

    csv_path, json_path = write_manifest(cache_dir, records)
    summary = {
        "total": len(records),
        "cached": sum(1 for r in records if r["status"] == "cached"),
        "skipped_existing": sum(1 for r in records if r["status"] == "skipped_existing"),
        "failed": sum(1 for r in records if r["status"] == "failed"),
        "cache_dir": str(cache_dir),
        "manifest_csv": str(csv_path),
        "manifest_json": str(json_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute decoded audio cache")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=SPLIT_CSV_PATH,
        help="Path to damp_sag_splits.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=AUDIO_CACHE_DIR,
        help="Directory for cached waveform .pt files",
    )
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument(
        "--overwrite",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Overwrite existing cache files (default: false)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only N files")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Reserved for future parallel precompute (currently sequential)",
    )
    parser.add_argument(
        "--fail-fast",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Stop on first failure (default: true)",
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Continue after failures and report summary at end",
    )
    args = parser.parse_args()

    if args.num_workers != 1:
        print(f"NOTE: --num-workers {args.num_workers} ignored; using sequential precompute.")

    fail_fast = args.fail_fast and not args.no_fail_fast

    if not args.split_csv.is_file():
        raise FileNotFoundError(f"Split CSV not found: {args.split_csv}")

    summary = precompute_audio_cache(
        split_csv=args.split_csv,
        cache_dir=args.cache_dir,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
        limit=args.limit,
        fail_fast=fail_fast,
    )

    print("\nAudio cache precompute complete")
    print("-----------------------------")
    print(f"total:            {summary['total']}")
    print(f"cached:           {summary['cached']}")
    print(f"skipped_existing: {summary['skipped_existing']}")
    print(f"failed:           {summary['failed']}")
    print(f"cache_dir:        {summary['cache_dir']}")
    print(f"manifest_csv:     {summary['manifest_csv']}")
    print(f"manifest_json:    {summary['manifest_json']}")

    if summary["failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
