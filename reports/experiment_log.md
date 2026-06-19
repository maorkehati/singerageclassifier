# Experiment Log

## Infrastructure test: dummy GPU training

Purpose:
Verify that the code can be developed locally, synced through GitHub, pulled on a GPU machine, and run there.

Expected command:
`python src/dummy_gpu_train.py --config configs/dummy_gpu.yaml`

Expected output:
A successful CUDA-only dummy training run and a metrics file in `experiments/dummy_gpu/metrics.json`.

## Infrastructure test: SLURM dummy GPU job

Purpose:
Verify the full work-side pipeline end to end:

- Local development on the home machine
- GitHub ZIP download and Windows sync into the shared filesystem
- Virtual environment setup on Linux (`scripts/setup_work_venv.sh`)
- SLURM job submission through `mem-ans1.transchip.com`
- CUDA execution on GPU compute node `gpu1`
- Metrics written to `experiments/dummy_gpu/metrics.json`

Expected submit command:
`/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier/slurm/submit_dummy_gpu.sh`

Expected output:
SLURM job completes successfully; logs appear under `slurm/logs/` and metrics under `experiments/dummy_gpu/metrics.json`.
