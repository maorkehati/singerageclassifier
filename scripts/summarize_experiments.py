"""Summarize controlled experiment results into markdown and CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from Sandbox.singerclassifier.experiments import EXPERIMENT_METADATA


def _load_json_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_summary_rows(
    experiments_root: Path,
    run_names: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for run_name in run_names:
        run_dir = experiments_root / run_name
        meta = EXPERIMENT_METADATA.get(run_name, {"main_change": "", "notes": ""})
        val_metrics = _load_json_metrics(run_dir / "val_metrics.json")
        test_metrics = _load_json_metrics(run_dir / "test_metrics.json")

        def fmt(metrics: dict[str, Any], key: str) -> str:
            if key not in metrics:
                return "MISSING"
            return f"{float(metrics[key]):.4f}"

        rows.append(
            {
                "run_name": run_name,
                "main_change": meta.get("main_change", ""),
                "val_accuracy": fmt(val_metrics, "accuracy"),
                "val_macro_f1": fmt(val_metrics, "macro_f1"),
                "val_balanced_accuracy": fmt(val_metrics, "balanced_accuracy"),
                "test_accuracy": fmt(test_metrics, "accuracy"),
                "test_macro_f1": fmt(test_metrics, "macro_f1"),
                "test_balanced_accuracy": fmt(test_metrics, "balanced_accuracy"),
                "notes": meta.get("notes", ""),
            }
        )

    return rows


def rows_to_markdown(rows: list[dict[str, str]]) -> str:
    headers = [
        "run_name",
        "main_change",
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
        "test_accuracy",
        "test_macro_f1",
        "test_balanced_accuracy",
        "notes",
    ]
    lines = [
        "# Phase 6 Experiment Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines) + "\n"


def save_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = build_summary_rows(args.experiments_root, args.run_names)
    markdown = rows_to_markdown(rows)

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown, encoding="utf-8")
    save_csv(rows, args.output_csv)

    print(markdown)
    print(f"Saved markdown summary to: {args.output_markdown}")
    print(f"Saved CSV summary to: {args.output_csv}")


if __name__ == "__main__":
    main()
