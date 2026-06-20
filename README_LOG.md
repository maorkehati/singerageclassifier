# Assignment Progress Log

## 1. Project workflow and GPU execution validated

We confirmed the development and execution workflow: code is developed locally, synced to the shared work filesystem, and executed through the Linux/GPU environment. A CUDA smoke test and dummy PyTorch training job successfully ran on the H100 GPU.

This matters because the assignment requires training a deep-learning model, and we now have a validated path for running GPU jobs and collecting experiment outputs.

Relevant paths/scripts:
- `/home/maork/Projects/rad_sandbox/Sandbox/singerclassifier`
- `/home/maork/Projects/rad_sandbox/Sandbox/data/DAMP-S-AG-partial/DAMP-S-AG`
- `slurm/`
- `experiments/`

## 2. Dataset inspection implemented

We implemented the Phase 1 data inspection script for the partial DAMP-S-AG dataset. The script loads `amazing_grace.tsv`, matches rows to `.m4a` files, computes singer age from `birth_year` and `creation_timestamp`, filters unusable examples, and reports age, demographic, account, and candidate bucket statistics.

This matters because the age-bucket design and train/validation/test split strategy should be based on the actual metadata distribution rather than assumptions. It also verifies that the dataset is accessible and usable from the work Linux/GPU environment.

Relevant files:
- `scripts/inspect_data.py`
- `singerclassifier/data.py`
- `singerclassifier/utils.py`
- `experiments/data_inspection/data_summary.json`
