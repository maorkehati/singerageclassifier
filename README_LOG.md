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
- `data/processed/damp_sag_splits.csv`
- `experiments/data_inspection/split_summary.json`

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
