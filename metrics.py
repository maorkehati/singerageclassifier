"""Classification metrics and reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics for age-bucket predictions."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    labels = list(range(len(class_names))) if class_names else None

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    support = {}

    if class_names:
        for name in class_names:
            if name in report:
                per_class_precision[name] = float(report[name]["precision"])
                per_class_recall[name] = float(report[name]["recall"])
                per_class_f1[name] = float(report[name]["f1-score"])
                support[name] = int(report[name]["support"])

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_arr, y_pred_arr)
        ),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "support": support,
        "confusion_matrix": confusion_matrix(
            y_true_arr,
            y_pred_arr,
            labels=labels,
        ).tolist(),
        "classification_report": report,
    }

    if y_prob is not None:
        metrics["probabilities_shape"] = list(np.asarray(y_prob).shape)

    return metrics


def save_classification_report(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Save a human-readable classification report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Classification Report",
        "=====================",
        f"accuracy: {metrics.get('accuracy', 0.0):.4f}",
        f"macro_f1: {metrics.get('macro_f1', 0.0):.4f}",
        f"balanced_accuracy: {metrics.get('balanced_accuracy', 0.0):.4f}",
        "",
    ]

    class_names = list(metrics.get("per_class_f1", {}).keys())
    if class_names:
        lines.append("Per-class metrics")
        lines.append("-----------------")
        for name in class_names:
            lines.append(
                f"{name}: "
                f"precision={metrics['per_class_precision'][name]:.4f}, "
                f"recall={metrics['per_class_recall'][name]:.4f}, "
                f"f1={metrics['per_class_f1'][name]:.4f}, "
                f"support={metrics['support'][name]}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_confusion_matrix_plot(
    confusion_matrix_values: Any,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Save a confusion matrix image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(confusion_matrix_values)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
