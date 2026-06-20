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

## Running Phase 6 experiments

Source-controlled code and sweep templates live in the repo. The raw dataset and persistent generated artifacts live outside the repo under:

```text
/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier
```

This includes the split CSV, generated concrete configs, sweep manifest, and decoded-audio cache. Keeping these outside the synced repo folder prevents repo syncs from deleting experiment prerequisites.

### Phase 6 workflow

1. Precompute persistent decoded-audio cache:

```bash
cd /home/maork/Projects/rad_sandbox/Sandbox/singerclassifier
bash scripts/precompute_phase6_cache.sh
```

Optional smoke run (5 files):

```bash
bash scripts/precompute_phase6_cache.sh --limit 5
```

2. Submit sweep:

```bash
bash scripts/remote_submit_phase6.sh 1
```

3. Check status/logs:

```bash
bash scripts/remote_status.sh
bash scripts/remote_tail_logs.sh 100
```

4. Summarize:

```bash
bash scripts/remote_summarize_phase6.sh
```

Cancel a submitted sweep (optional):

```bash
bash scripts/remote_cancel_phase6.sh <job_id>
```

The submit script validates the cache but does not build it by default, so submission should be fast. The workflow sends non-interactive commands to `mem-ans1`; there is no need to open an interactive SSH shell.

### Audio cache

Decoded mono 22.05 kHz waveform tensors are cached outside the repo:

`/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier/cache/audio_22050_mono`

This prevents repeated `.m4a` decoding through ffmpeg during every training epoch. The cache is persistent across repo syncs.

## Progress Log

See `README_LOG.md` for a brief numbered log of assignment milestones and implementation decisions.

## Next steps

After confirming the dummy GPU run works on your GPU machine, proceed with the real singing-age classifier implementation using the assignment dataset.
