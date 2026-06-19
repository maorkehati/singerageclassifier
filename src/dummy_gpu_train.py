"""Dummy GPU training experiment for infrastructure verification."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available. This script requires a CUDA-capable GPU.",
            file=sys.stderr,
        )
        sys.exit(1)
    return torch.device("cuda")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_synthetic_dataset(
    num_samples: int,
    input_dim: int,
    num_classes: int,
    device: torch.device,
) -> TensorDataset:
    features = torch.randn(num_samples, input_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)
    return TensorDataset(features, labels)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def train(config: dict, device: torch.device) -> dict:
    set_seed(config["seed"])

    dataset = create_synthetic_dataset(
        num_samples=config["num_samples"],
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        device=device,
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = MLP(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    final_loss = 0.0
    final_accuracy = 0.0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for features, labels in loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_correct += (logits.argmax(dim=1) == labels).sum().item()
            epoch_total += batch_size

        final_loss = epoch_loss / epoch_total
        final_accuracy = epoch_correct / epoch_total
        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"loss: {final_loss:.4f} | accuracy: {final_accuracy:.4f}"
        )

    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "epochs": config["epochs"],
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy GPU training experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = require_cuda()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(config, device)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
