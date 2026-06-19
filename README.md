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

## Work SLURM workflow

This repo is deployed on the work shared filesystem at:

- Windows: `\\mars\raid\users\maork\Projects\rad_sandbox\Sandbox\singerclassifier`
- Linux: `/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier`

### A. On home machine

- Edit code in Cursor
- Commit and push to GitHub

### B. On work Windows machine

- Download the GitHub ZIP to:
  `C:\Users\maork\Downloads\singerageclassifier-main.zip`
- Run the PowerShell sync script that copies it to:
  `\\mars\raid\users\maork\Projects\rad_sandbox\Sandbox\singerclassifier`

### C. On Linux / shared filesystem — first-time setup

```bash
/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/scripts/setup_work_venv.sh
```

Creates `.venv`, installs dependencies, and runs import/compile checks (no CUDA required).

### D. Submit dummy GPU job

```bash
/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/submit_dummy_gpu.sh
```

Or from any machine with SSH access:

```bash
ssh mem-ans1.transchip.com "sbatch /home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/dummy_gpu.sbatch"
```

### E. Check status

```bash
/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/status.sh
```

### F. Cancel a job

```bash
/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/cancel.sh <job_id>
```

### G. Logs

Job stdout/stderr:

```
/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/logs/
```

Training metrics (after a successful run):

```
/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/experiments/dummy_gpu/metrics.json
```

## Next steps

After confirming the dummy GPU run works on your GPU machine, proceed with the real singing-age classifier implementation using the assignment dataset.
