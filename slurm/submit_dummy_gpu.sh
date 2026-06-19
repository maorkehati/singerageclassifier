#!/usr/bin/env bash
set -euo pipefail

REMOTE_CMD='cd /home/maork/Projects/rad_sandbox/Sandbox/singerclassifier && mkdir -p slurm/logs experiments/dummy_gpu && sbatch slurm/dummy_gpu.sbatch'
SSH_CMD="ssh mem-ans1.transchip.com \"${REMOTE_CMD}\""

echo "Running: ${SSH_CMD}"
ssh mem-ans1.transchip.com "${REMOTE_CMD}"
