#!/usr/bin/env bash
set -euo pipefail

REMOTE_CMD='cd /home/maork/Projects/rad_sandbox/Sandbox/singerclassifier && mkdir -p slurm/logs experiments && sbatch slurm/run_experiment.sbatch phase6'
SSH_CMD="ssh mem-ans1.transchip.com \"${REMOTE_CMD}\""

echo "Running: ${SSH_CMD}"
ssh mem-ans1.transchip.com "${REMOTE_CMD}"
