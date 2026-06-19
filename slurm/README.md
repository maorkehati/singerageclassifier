# SLURM job scripts

Scripts for submitting the dummy GPU infrastructure test on the work cluster via `mem-ans1.transchip.com`.

| Script | Purpose |
|--------|---------|
| `dummy_gpu.sbatch` | SLURM batch script: smoke test + dummy training on `gpu1` |
| `submit_dummy_gpu.sh` | Submit the batch job over SSH (run from any directory) |
| `status.sh` | Show your SLURM queue |
| `cancel.sh` | Cancel a job by ID |

## Paths (Linux)

- Work root: `/home/maork/Projects/rad_sandbox`
- Repo: `/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier`
- Logs: `/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/logs/`

## First-time setup

```bash
/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/scripts/setup_work_venv.sh
```

## Submit

```bash
/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/submit_dummy_gpu.sh
```

Or from WSL/home:

```bash
ssh mem-ans1.transchip.com "sbatch /home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/dummy_gpu.sbatch"
```
