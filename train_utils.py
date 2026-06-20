"""Training utilities for reproducible experiments."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


DEFAULT_CLASS_NAMES = ["under_25", "age_25_34", "age_35_plus"]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str | None = None) -> torch.device:
    """Resolve the compute device."""
    if device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def compute_class_weights_from_split_csv(
    split_csv: str | Path,
    split: str = "train",
    num_classes: int = 3,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from a split CSV."""
    df = pd.read_csv(split_csv)
    train_df = df[df["split"] == split]
    if train_df.empty:
        raise ValueError(f"No rows found for split '{split}' in {split_csv}")

    counts = train_df["age_bucket_id"].value_counts().sort_index()
    total_samples = len(train_df)
    weights = []

    for class_id in range(num_classes):
        class_count = int(counts.get(class_id, 0))
        if class_count == 0:
            weights.append(0.0)
        else:
            weights.append(total_samples / (num_classes * class_count))

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if weight_tensor.sum() > 0:
        weight_tensor = weight_tensor / weight_tensor.mean()
    return weight_tensor


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def get_git_commit_hash() -> str | None:
    """Return the current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class EarlyStopping:
    """Track validation macro-F1 with patience-based early stopping."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-6,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("-inf")
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
        self.improved = False

    def step(self, metric: float, loss: float, epoch: int) -> bool:
        """Update state with the latest validation metrics."""
        self.improved = False
        improved = False

        if metric > self.best_metric + self.min_delta:
            improved = True
        elif abs(metric - self.best_metric) <= self.min_delta and loss < self.best_loss:
            improved = True

        if improved:
            self.best_metric = metric
            self.best_loss = loss
            self.best_epoch = epoch
            self.counter = 0
            self.improved = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.improved


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract model input and labels from a dataloader batch."""
    if isinstance(batch, dict):
        x = batch["input"]
        y = batch["label"]
    else:
        x, y = batch

    if not torch.is_tensor(y):
        y = torch.tensor(y)
    return x, y.long()


def run_inference(
    model: torch.nn.Module,
    dataloader: Any,
    criterion: torch.nn.Module | None,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """Run deterministic inference and compute metrics."""
    from .metrics import compute_classification_metrics

    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * y.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_labels.extend(y.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

    metrics = compute_classification_metrics(
        all_labels,
        all_preds,
        y_prob=all_probs,
        class_names=class_names,
    )
    num_samples = len(all_labels)
    metrics["loss"] = total_loss / max(num_samples, 1) if criterion else 0.0
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Any,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_names: list[str],
) -> dict[str, float]:
    """Train for one epoch and return metrics."""
    from .metrics import compute_classification_metrics

    model.train()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch in dataloader:
        x, y = unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        all_labels.extend(y.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    metrics = compute_classification_metrics(
        all_labels,
        all_preds,
        class_names=class_names,
    )
    num_samples = len(all_labels)
    metrics["loss"] = total_loss / max(num_samples, 1)
    return metrics


def evaluate_one_epoch(
    model: torch.nn.Module,
    dataloader: Any,
    criterion: torch.nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """Evaluate for one epoch."""
    return run_inference(model, dataloader, criterion, device, class_names)


def run_multicrop_inference(
    model: torch.nn.Module,
    config: dict[str, Any],
    split: str,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """Evaluate with deterministic multi-crop logit averaging."""
    from .audio import crop_from_start, deterministic_crop_starts, load_audio
    from .features import LogMelSpectrogram
    from .metrics import compute_classification_metrics

    data_cfg = config["data"]
    num_crops = int(data_cfg.get("eval_num_crops", 1))
    split_csv = data_cfg["split_csv"]
    sample_rate = data_cfg.get("sample_rate", 22050)
    duration_sec = data_cfg.get("duration_sec", 15.0)
    target_samples = int(sample_rate * duration_sec)

    df = pd.read_csv(split_csv)
    split_df = df[df["split"] == split].reset_index(drop=True)
    if split_df.empty:
        raise ValueError(f"No rows found for split '{split}' in {split_csv}")

    feature_extractor = LogMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=data_cfg.get("n_fft", 1024),
        hop_length=data_cfg.get("hop_length", 512),
        n_mels=data_cfg.get("n_mels", 80),
        f_min=data_cfg.get("f_min", 50.0),
        f_max=data_cfg.get("f_max", 8000.0),
        normalize=True,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []
    total_loss = 0.0

    with torch.no_grad():
        for row_idx, row in split_df.iterrows():
            audio_path = Path(str(row["audio_path"]))
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            waveform, _ = load_audio(
                audio_path,
                target_sample_rate=sample_rate,
                mono=True,
            )
            starts = deterministic_crop_starts(
                waveform.shape[-1],
                target_samples,
                num_crops,
            )

            logits_list = []
            for start in starts:
                crop = crop_from_start(waveform, start, target_samples)
                mel = feature_extractor(crop).unsqueeze(0).to(device)
                logits_list.append(model(mel))

            avg_logits = torch.stack(logits_list, dim=0).mean(dim=0).squeeze(0)
            label = torch.tensor(int(row["age_bucket_id"]), device=device)
            total_loss += criterion(avg_logits.unsqueeze(0), label.unsqueeze(0)).item()

            probs = torch.softmax(avg_logits, dim=0)
            pred = int(avg_logits.argmax().item())
            all_labels.append(int(label.item()))
            all_preds.append(pred)
            all_probs.append(probs.detach().cpu().tolist())

    metrics = compute_classification_metrics(
        all_labels,
        all_preds,
        y_prob=all_probs,
        class_names=class_names,
    )
    metrics["loss"] = total_loss / max(len(all_labels), 1)
    metrics["eval_num_crops"] = num_crops
    return metrics


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
