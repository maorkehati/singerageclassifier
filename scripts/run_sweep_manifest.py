"""Run experiments listed in a sweep manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from Sandbox.singerclassifier.experiments import is_majority_baseline_config
from Sandbox.singerclassifier.scripts.evaluate import evaluate_run
from Sandbox.singerclassifier.scripts.run_majority_baseline import run_majority_baseline
from Sandbox.singerclassifier.scripts.train import train
from Sandbox.singerclassifier.sweep import read_manifest_csv, write_manifest_csv
from Sandbox.singerclassifier.train_utils import load_yaml


def should_skip_row(row: dict, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    test_metrics = Path(row["run_dir"]) / "test_metrics.json"
    return test_metrics.is_file()


def run_manifest_row(row: dict) -> None:
    config = load_yaml(row["config_path"])
    if is_majority_baseline_config(config):
        run_majority_baseline(config)
    else:
        run_dir = train(config)
        evaluate_run(run_dir, split="test")


def run_sweep_manifest(
    manifest_path: Path,
    index: int | None = None,
    family: str | None = None,
    skip_existing: bool = False,
    continue_on_error: bool = False,
) -> list[dict]:
    rows = read_manifest_csv(manifest_path)
    selected = rows

    if index is not None:
        selected = [row for row in rows if row["index"] == index]
        if not selected:
            raise ValueError(f"No manifest row with index={index}")

    if family is not None:
        selected = [row for row in selected if row["family"] == family]
        if not selected:
            raise ValueError(f"No manifest rows for family={family}")

    failures = 0
    for row in selected:
        if should_skip_row(row, skip_existing):
            row["status"] = "skipped"
            print(f"Skipping existing run: {row['run_name']}")
            continue

        print(f"\n=== Starting run {row['index']}: {row['run_name']} ({row['family']}) ===")
        row["status"] = "running"
        write_manifest_csv(manifest_path, rows)

        try:
            run_manifest_row(row)
            row["status"] = "completed"
            print(f"=== Completed run: {row['run_name']} ===")
        except Exception as exc:
            row["status"] = "failed"
            failures += 1
            print(f"=== Failed run: {row['run_name']}: {exc} ===")
            if not continue_on_error:
                write_manifest_csv(manifest_path, rows)
                raise

        write_manifest_csv(manifest_path, rows)

    if failures and continue_on_error:
        print(f"\n{failures} run(s) failed (continue-on-error enabled).")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sweep manifest experiments")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    run_sweep_manifest(
        manifest_path=args.manifest,
        index=args.index,
        family=args.family,
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()
