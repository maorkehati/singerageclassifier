# Singer Age Classifier

Infrastructure and proof-of-work setup for the singing age classifier take-home assignment. This repository currently contains a minimal PyTorch project with a dummy GPU training experiment to verify that the development workflow works across machines.

## Purpose

Before implementing the real singing-age classifier, this repo validates:

- Python project structure and dependencies
- CUDA GPU availability on a target machine
- Git-based sync between a local WSL environment and a separate GPU work machine

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your GPU machine. See the [PyTorch install guide](https://pytorch.org/get-started/locally/) if the default `pip install torch` does not detect your GPU.

## Running the infrastructure tests

**CUDA smoke test** — quick check that PyTorch sees your GPU:

```bash
python scripts/smoke_gpu.py
```

**Dummy GPU training** — small synthetic classification run that requires CUDA:

```bash
python src/dummy_gpu_train.py --config configs/dummy_gpu.yaml
```

On success, metrics are written to `experiments/dummy_gpu/metrics.json`.

Both scripts exit with a clear error if CUDA is unavailable.

## Git exclusions

The following are intentionally excluded from version control:

- `data/` — real assignment datasets (not downloaded yet)
- `checkpoints/` — model weights
- `experiments/` — run outputs (metrics, logs, artifacts)

Only placeholder `.gitkeep` files are tracked in those directories.

## Next steps

After confirming the dummy GPU run works on your GPU machine, proceed with the real singing-age classifier implementation using the assignment dataset.
