"""Run controlled experiment configs sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from Sandbox.singerclassifier.experiments import (
    PHASE6_CONFIGS,
    get_run_dir,
    is_majority_baseline_config,
    resolve_config_path,
    test_metrics_path,
)
from Sandbox.singerclassifier.scripts.evaluate import evaluate_run
from Sandbox.singerclassifier.scripts.run_majority_baseline import run_majority_baseline
from Sandbox.singerclassifier.scripts.train import train
from Sandbox.singerclassifier.train_utils import load_yaml, save_json


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_config_list(
    config_paths: list[str] | None,
    experiment_set: str | None,
) -> list[Path]:
    root = repo_root()
    if experiment_set == "phase6":
        return [resolve_config_path(path, root) for path in PHASE6_CONFIGS]
    if not config_paths:
        raise ValueError("Provide --configs or --experiment-set phase6")
    return [resolve_config_path(path, root) for path in config_paths]


def run_experiments(
    config_paths: list[Path],
    continue_on_error: bool = False,
    skip_existing: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {"experiments": []}

    for config_path in config_paths:
        config = load_yaml(config_path)
        run_name = config["run_name"]
        run_dir = get_run_dir(config)
        entry: dict[str, Any] = {
            "run_name": run_name,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "status": "pending",
            "error": None,
        }
        manifest["experiments"].append(entry)

        if skip_existing and test_metrics_path(config).is_file():
            entry["status"] = "skipped"
            print(f"Skipping existing experiment: {run_name}")
            continue

        print(f"\n=== Starting experiment: {run_name} ===")
        entry["status"] = "running"

        try:
            if is_majority_baseline_config(config):
                run_majority_baseline(config)
            else:
                completed_run_dir = train(config)
                evaluate_run(completed_run_dir, split="test")

            entry["status"] = "completed"
            print(f"=== Completed experiment: {run_name} ===")
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            print(f"=== Failed experiment: {run_name}: {exc} ===")
            if not continue_on_error:
                break

    if manifest_path is None and config_paths:
        first_config = load_yaml(config_paths[0])
        manifest_path = Path(first_config["output"]["root_dir"]) / "phase6_manifest.json"

    if manifest_path is not None:
        save_json(manifest, manifest_path)
        print(f"\nSaved experiment manifest to: {manifest_path}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled experiments")
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--experiment-set", type=str, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--manifest-path", type=Path, default=None)
    args = parser.parse_args()

    config_paths = resolve_config_list(args.configs, args.experiment_set)
    manifest = run_experiments(
        config_paths=config_paths,
        continue_on_error=args.continue_on_error,
        skip_existing=args.skip_existing,
        manifest_path=args.manifest_path,
    )

    failed = [e for e in manifest["experiments"] if e["status"] == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} experiment(s) failed.")


if __name__ == "__main__":
    main()
