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

## 3. Leakage-safe train/validation/test splits implemented

We implemented Phase 2 split creation for the usable DAMP-S-AG subset. The split script assigns the selected 3-class age buckets and creates train, validation, and test sets using an account-level stratified splitting strategy.

This matters because splitting by `account_id` prevents the same singer from appearing in multiple splits, reducing singer-identity leakage and making the evaluation more meaningful. Stratifying by the account-level modal age bucket keeps the age-class distribution approximately balanced across splits.

Chosen split algorithm:
- Account-level split.
- Stratification label: modal `age_bucket_id` per `account_id`.
- First split: train vs temporary set.
- Second split: validation vs test from the temporary set.
- Default ratios: 70% train, 15% validation, 15% test.
- Seed: 42.

Relevant files:
- `scripts/prepare_splits.py`
- `singerclassifier/splits.py`
- `data/processed/damp_sag_splits.csv` (legacy; canonical path is under `Sandbox/data/singerclassifier/processed/`)
- `experiments/data_inspection/split_summary.json` (legacy; canonical path is under `Sandbox/data/singerclassifier/data_inspection/`)

## 4. Audio preprocessing implemented

We implemented the Phase 3 audio preprocessing pipeline. The dataset now loads `.m4a` recordings from the leakage-safe split CSV, converts each recording to mono 22.05 kHz audio, crops or pads it to a fixed duration, and transforms it into a normalized log-mel spectrogram suitable for a CNN trained from scratch.

This matters because the assignment requires a deep-learning model trained from raw singing performances without pretrained audio models. Log-mel spectrograms provide a stable time-frequency representation while keeping the model architecture simple and fully trainable from scratch.

Chosen preprocessing:
- 22.05 kHz mono audio.
- 15-second fixed-length crops.
- Random crop for training.
- Center crop for validation/test.
- 80-bin log-mel spectrogram.
- Per-sample normalization.

Relevant files:
- `singerclassifier/audio.py`
- `singerclassifier/features.py`
- `singerclassifier/data.py`
- `scripts/smoke_audio_preprocessing.py`

## 5. Scratch CNN model implemented

We implemented the Phase 4 model architecture: a compact 2D convolutional neural network over log-mel spectrogram inputs. The model uses four convolutional blocks followed by adaptive average pooling and a small classifier head.

This matters because the assignment requires a deep-learning model built from scratch without pretrained audio models or pretrained weights. The architecture is intentionally small for the limited partial dataset while still being expressive enough to learn local time-frequency patterns from singing audio.

Chosen architecture:
- Input: normalized log-mel spectrograms with shape `[batch, 1, n_mels, time]`.
- Four Conv2D blocks with BatchNorm and ReLU.
- Max pooling in the early blocks.
- Adaptive average pooling before classification.
- Linear classifier head producing 3 age-bucket logits.

Relevant files:
- `singerclassifier/models.py`
- `scripts/smoke_model.py`

## 6. Training and evaluation pipeline implemented

We implemented the Phase 5 training setup for reproducible model experiments. The pipeline now supports YAML-configured runs, deterministic seeding, AdamW optimization, cross-entropy loss, optional class-weighted loss, validation-based checkpointing, and saved metrics/artifacts for comparison.

This matters because the assignment requires training a model, evaluating performance, and comparing several iterations. The training pipeline makes each experiment reproducible and stores the outputs needed for the final report.

Default training setup:
- AdamW optimizer.
- Cross-entropy loss.
- Optional class-weighted cross-entropy.
- Early stopping based on validation macro-F1.
- Saved best checkpoint.
- Accuracy, macro-F1, balanced accuracy, per-class metrics, and confusion matrix.

Relevant files:
- `singerclassifier/train_utils.py`
- `singerclassifier/metrics.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `configs/cnn_basic.yaml`
- `configs/cnn_balanced.yaml`

## 7. Controlled experiment runner implemented

We implemented the Phase 6 experiment framework for running and comparing controlled iterations. The framework defines the planned baseline and CNN experiments through configs, runs each experiment into its own output directory, evaluates checkpoints on the test split, and summarizes all results into a markdown/CSV comparison table.

This matters because the assignment asks for several iterations, performance comparison, and explanation of the differences. The experiment framework makes the iteration story explicit and reproducible rather than relying on ad hoc training commands.

Planned controlled experiments:
- `majority_baseline`: trivial reference point.
- `cnn_basic`: first scratch CNN.
- `cnn_balanced`: class-weighted loss.
- `cnn_augmented`: light augmentation.
- `cnn_augmented_multicrop`: multi-crop evaluation.

Relevant files:
- `configs/majority_baseline.yaml`
- `configs/cnn_basic.yaml`
- `configs/cnn_balanced.yaml`
- `configs/cnn_augmented.yaml`
- `configs/cnn_augmented_multicrop.yaml`
- `scripts/run_majority_baseline.py`
- `scripts/run_experiments.py`
- `scripts/summarize_experiments.py`
- `experiments/phase6_summary.md`

## 8. Controlled sweep framework implemented

We replaced the ad hoc experiment execution with a controlled sweep framework. A single sweep specification now defines the planned experiment families, generates concrete run configs, creates a manifest, supports local or SLURM execution, and summarizes all completed runs.

This matters because the assignment asks for several meaningful iterations and comparison of performance. The sweep framework makes each iteration reproducible while keeping the search space intentionally small and interpretable rather than exhaustive.

Sweep design:
- `majority_baseline`: one deterministic baseline.
- `cnn_basic`: small optimization/dropout sweep for the first scratch CNN.
- `cnn_balanced`: same sweep with class-weighted loss.
- `cnn_augmented`: light/medium augmentation variants.
- `cnn_augmented_multicrop`: multi-crop evaluation variants.

Relevant files:
- `configs/phase6_sweeps.yaml`
- `scripts/generate_sweep_configs.py`
- `scripts/run_sweep_manifest.py`
- `scripts/summarize_sweep.py`
- `slurm/phase6_sweep_array.sbatch`
- `slurm/submit_phase6_sweep.sh`

## 9. Friendly remote SLURM submission added

We updated the experiment workflow so Phase 6 sweeps can be launched from the work Linux machine with a single local command. The local helper script sends the SLURM submission command non-interactively to `mem-ans1`, where `sbatch` is available, while logs and experiment outputs remain on the shared filesystem.

This matters because the work Linux machine can access the project files but does not provide `sbatch` directly. The new remote submission helpers make the controlled sweep easier to run without manually opening an SSH session or writing one-off SLURM commands.

Relevant files:
- `scripts/remote_submit_phase6.sh`
- `scripts/remote_status.sh`
- `scripts/remote_tail_logs.sh`
- `scripts/remote_summarize_phase6.sh`
- `scripts/remote_cancel_phase6.sh`
- `slurm/submit_phase6_sweep.sh`
- `slurm/phase6_sweep_array.sbatch`

## 10. Persistent generated artifacts moved outside the repo

We moved generated artifacts required for experiment execution outside the synced repo folder. The split CSV, generated sweep configs, and sweep manifests now live under `/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier`.

This matters because the repo sync procedure can overwrite the `singerclassifier` folder. Keeping generated experiment prerequisites in the shared data area prevents split files and generated configs from disappearing after code updates.

Persistent artifact paths:
- Split CSV: `/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier/processed/damp_sag_splits.csv`
- Generated configs: `/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier/generated_configs/phase6`
- Sweep manifest: `/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier/manifests/phase6_sweep_manifest.csv`

Relevant files:
- `configs/phase6_sweeps.yaml`
- `scripts/prepare_splits.py`
- `scripts/generate_sweep_configs.py`
- `scripts/run_sweep_manifest.py`
- `slurm/submit_phase6_sweep.sh`
- `slurm/phase6_sweep_array.sbatch`

## 11. Audio decoding switched to ffmpeg executable fallback

We fixed the `.m4a` audio decoding failure by adding an ffmpeg subprocess backend to the audio loader. The environment has a static ffmpeg executable but does not expose FFmpeg shared libraries required by TorchCodec, so relying on `torchaudio.load` alone is not reliable for `.m4a` files.

This matters because all Phase 6 CNN experiments depend on loading `.m4a` DAMP-S-AG recordings inside the PyTorch dataloader. The new loader uses the available ffmpeg executable and avoids the missing shared-library dependency.

Relevant files:
- `singerclassifier/audio.py`
- `scripts/smoke_audio_preprocessing.py`
- `slurm/submit_phase6_sweep.sh`

## 12. Persistent decoded-audio cache added

We added a persistent audio cache outside the synced repo folder. The cache stores decoded mono 22.05 kHz waveform tensors under `/home/maork/Projects/rad_sandbox/Sandbox/data/singerclassifier/cache/audio_22050_mono`.

This matters because repeatedly decoding `.m4a` files through ffmpeg inside every dataloader epoch is slow and can leave the GPU underutilized. Caching decoded waveforms removes the repeated decode bottleneck while preserving random crops, multi-crop evaluation, and waveform-level augmentation.

Relevant files:
- `scripts/precompute_audio_cache.py`
- `singerclassifier/audio.py`
- `singerclassifier/data.py`
- `configs/phase6_sweeps.yaml`
- `slurm/submit_phase6_sweep.sh`
