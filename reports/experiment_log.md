# Experiment Log

## Infrastructure test: dummy GPU training

Purpose:
Verify that the code can be developed locally, synced through GitHub, pulled on a GPU machine, and run there.

Expected command:
`python src/dummy_gpu_train.py --config configs/dummy_gpu.yaml`

Expected output:
A successful CUDA-only dummy training run and a metrics file in `experiments/dummy_gpu/metrics.json`.
