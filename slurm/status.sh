#!/usr/bin/env bash
set -euo pipefail

ssh mem-ans1.transchip.com "squeue -u \$USER"
