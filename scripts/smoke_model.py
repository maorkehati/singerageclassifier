"""Smoke test for scratch CNN model construction and forward/backward passes."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from Sandbox.singerclassifier.data import build_dataloader
from Sandbox.singerclassifier.models import build_model, describe_model


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_has_gradients(model: nn.Module) -> None:
    has_grad = any(
        param.grad is not None and torch.isfinite(param.grad).any()
        for param in model.parameters()
    )
    if not has_grad:
        raise AssertionError("Backward pass did not produce any parameter gradients.")


def run_dummy_smoke_test(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    n_mels: int,
    time_frames: int,
    num_classes: int,
) -> None:
    model = model.to(device)
    summary = describe_model(model)

    print("Model summary")
    print("-------------")
    print(f"class name: {summary['model_class']}")
    print(f"total parameters: {summary['total_parameters']}")
    print(f"trainable parameters: {summary['trainable_parameters']}")

    x = torch.randn(batch_size, 1, n_mels, time_frames, device=device)
    logits = model(x)

    print("\nForward pass")
    print("------------")
    print(f"input shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(logits.shape)}")
    print(f"output dtype: {logits.dtype}")

    expected = (batch_size, num_classes)
    if tuple(logits.shape) != expected:
        raise AssertionError(
            f"Expected output shape {expected}, got {tuple(logits.shape)}"
        )

    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    loss.backward()
    assert_has_gradients(model)

    print("\nBackward pass")
    print("-------------")
    print(f"loss: {loss.item():.4f}")
    print("gradients: OK")


def run_dataloader_smoke_test(
    model: nn.Module,
    device: torch.device,
    split_csv: str,
    batch_size: int,
    num_classes: int,
) -> None:
    model = model.to(device)
    loader = build_dataloader(
        split_csv=split_csv,
        split="train",
        batch_size=batch_size,
        num_workers=0,
    )

    batch_x, batch_y = next(iter(loader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    logits = model(batch_x)

    print("\nDataloader integration")
    print("----------------------")
    print(f"batch input shape: {tuple(batch_x.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"labels: {batch_y.tolist()}")

    expected = (batch_x.shape[0], num_classes)
    if tuple(logits.shape) != expected:
        raise AssertionError(
            f"Expected logits shape {expected}, got {tuple(logits.shape)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test SmallMelCNN model")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--time-frames", type=int, default=646)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--split-csv",
        type=str,
        default=None,
        help="Optional split CSV for dataloader integration smoke test",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = build_model(model_name="small_mel_cnn", num_classes=args.num_classes)

    run_dummy_smoke_test(
        model=model,
        device=device,
        batch_size=args.batch_size,
        n_mels=args.n_mels,
        time_frames=args.time_frames,
        num_classes=args.num_classes,
    )

    if args.split_csv:
        run_dataloader_smoke_test(
            model=model,
            device=device,
            split_csv=args.split_csv,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
        )

    print("\nModel smoke test passed.")


if __name__ == "__main__":
    main()
