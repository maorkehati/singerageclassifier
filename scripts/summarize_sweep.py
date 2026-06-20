"""Summarize sweep manifest runs and select best run per family."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from Sandbox.singerclassifier.sweep import FAMILY_DESCRIPTIONS, read_manifest_csv
from Sandbox.singerclassifier.train_utils import load_yaml
from Sandbox.singerclassifier.utils import EXPERIMENTS_ROOT_PATH, MANIFEST_CSV_PATH


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_value(metrics: dict[str, Any], key: str) -> str:
    if key not in metrics:
        return "MISSING"
    return f"{float(metrics[key]):.4f}"


def _config_value(config: dict[str, Any], *keys: str, default: str = "") -> str:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if isinstance(current, bool):
        return str(current).lower()
    return str(current)


def build_run_row(row: dict[str, Any]) -> dict[str, str]:
    run_dir = Path(row["run_dir"])
    config = load_yaml(run_dir / "config.yaml") if (run_dir / "config.yaml").is_file() else {}
    if not config and Path(row["config_path"]).is_file():
        config = load_yaml(row["config_path"])

    val_metrics = _load_json(run_dir / "val_metrics.json")
    test_metrics = _load_json(run_dir / "test_metrics.json")

    aug_profile = _config_value(config, "augmentation", "profile")
    if not aug_profile:
        aug_profile = _config_value(config, "sweep_parameters", "augmentation.profile")

    return {
        "family": row["family"],
        "run_name": row["run_name"],
        "learning_rate": _config_value(config, "training", "learning_rate"),
        "dropout": _config_value(config, "model", "dropout"),
        "weight_decay": _config_value(config, "training", "weight_decay"),
        "class_weighted_loss": _config_value(config, "training", "class_weighted_loss"),
        "augment_train": _config_value(config, "data", "augment_train"),
        "augmentation_profile": aug_profile,
        "eval_num_crops": _config_value(config, "data", "eval_num_crops", default="1"),
        "val_accuracy": _metric_value(val_metrics, "accuracy"),
        "val_macro_f1": _metric_value(val_metrics, "macro_f1"),
        "val_balanced_accuracy": _metric_value(val_metrics, "balanced_accuracy"),
        "test_accuracy": _metric_value(test_metrics, "accuracy"),
        "test_macro_f1": _metric_value(test_metrics, "macro_f1"),
        "test_balanced_accuracy": _metric_value(test_metrics, "balanced_accuracy"),
        "status": row.get("status", "unknown"),
    }


def select_best_by_family(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best_rows: list[dict[str, str]] = []
    families = sorted({row["family"] for row in rows})

    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        scored = []
        for row in family_rows:
            if row["val_macro_f1"] == "MISSING":
                scored.append((float("-inf"), row))
            else:
                scored.append((float(row["val_macro_f1"]), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0][1].copy()
        meta = FAMILY_DESCRIPTIONS.get(family, {})
        best["main_change"] = meta.get("main_change", "")
        best["notes"] = meta.get("notes", "")
        best_rows.append(best)

    return best_rows


def rows_to_markdown(title: str, rows: list[dict[str, str]]) -> str:
    if not rows:
        return f"# {title}\n\nNo runs found.\n"

    headers = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines) + "\n"


def save_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_sweep(manifest_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    manifest_rows = read_manifest_csv(manifest_path)
    if output_root is None:
        output_root = EXPERIMENTS_ROOT_PATH

    all_rows = [build_run_row(row) for row in manifest_rows]
    best_rows = select_best_by_family(all_rows)

    all_csv = output_root / "phase6_all_runs_summary.csv"
    all_md = output_root / "phase6_all_runs_summary.md"
    best_csv = output_root / "phase6_best_by_family.csv"
    best_md = output_root / "phase6_best_by_family.md"

    save_csv(all_rows, all_csv)
    all_md.write_text(rows_to_markdown("Phase 6 All Runs Summary", all_rows), encoding="utf-8")
    save_csv(best_rows, best_csv)
    best_md.write_text(rows_to_markdown("Phase 6 Best By Family", best_rows), encoding="utf-8")

    return {
        "all_runs_csv": str(all_csv),
        "all_runs_md": str(all_md),
        "best_by_family_csv": str(best_csv),
        "best_by_family_md": str(best_md),
        "all_rows": all_rows,
        "best_rows": best_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sweep manifest results")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_CSV_PATH,
        help="Path to sweep manifest CSV",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for summary files (default: repo experiments/)",
    )
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        output_root = EXPERIMENTS_ROOT_PATH

    result = summarize_sweep(args.manifest, output_root)

    print(Path(result["all_runs_md"]).read_text(encoding="utf-8"))
    print(Path(result["best_by_family_md"]).read_text(encoding="utf-8"))
    print(f"Saved all-runs summary: {result['all_runs_csv']}")
    print(f"Saved best-by-family summary: {result['best_by_family_csv']}")


if __name__ == "__main__":
    main()
