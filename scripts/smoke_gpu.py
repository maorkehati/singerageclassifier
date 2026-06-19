"""CUDA smoke test: verify PyTorch GPU availability and basic ops."""

import sys
import time

import torch


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available. This project requires a CUDA-capable GPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda")
    size = 4096
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Matrix multiply ({size}x{size}) elapsed: {elapsed:.4f}s")
    print("CUDA smoke test passed.")


if __name__ == "__main__":
    main()
