# Singing Voice Age Classification

This project builds a singing-voice age classifier using the partial DAMP-S-AG dataset. The system predicts one of three singer age buckets from an audio recording of “Amazing Grace.” The final model is a compact convolutional neural network trained from scratch on log-mel spectrograms, without using pretrained audio models or pretrained weights.

## Overview

| Item | Description |
|---|---|
| Input | Singing audio recordings (`.m4a`) |
| Output | One of three age buckets |
| Model | Scratch 2D CNN over log-mel spectrograms |
| Pretrained models | Not used |

The pipeline loads metadata and audio, converts fixed-length waveform crops to normalized log-mel features, trains a small CNN with optional class weighting and augmentation, and evaluates with standard classification metrics. Model selection uses validation macro-F1.

## Dataset

The project uses the partial DAMP-S-AG dataset: metadata in `amazing_grace.tsv` and corresponding `.m4a` recordings in `amazing_grace/`.

After filtering invalid metadata, missing audio, and out-of-range ages:

| Statistic | Count |
|---|---:|
| Raw metadata rows | 2,152 |
| Usable labeled recordings | 1,197 |
| Dropped invalid/missing birth year | 945 |
| Dropped invalid age | 10 |
| Missing audio among usable rows | 0 |

Age is computed from `birth_year` and `creation_timestamp`.

### Age buckets

Three classes were used because they are reasonably balanced for a small dataset:

| Bucket | Definition | Count |
|---|---|---:|
| `under_25` | age &lt; 25 | 378 |
| `age_25_34` | 25 ≤ age &lt; 35 | 407 |
| `age_35_plus` | age ≥ 35 | 412 |

## Split Methodology

Recordings are split into train, validation, and test sets with default ratios of **70% / 15% / 15%** and **seed 42**.

To reduce identity leakage, the dataset is split by `account_id`, not by individual recording. This prevents recordings from the same singer account from appearing in multiple splits. The account-level split is stratified by the modal age bucket of each account.

## Preprocessing

Audio preprocessing follows a fixed pipeline:

| Step | Setting |
|---|---|
| Source format | `.m4a` |
| Waveform | Mono |
| Sample rate | 22.05 kHz |
| Segment length | 15-second crop or zero-pad |
| Features | 80-bin log-mel spectrogram |
| Normalization | Per-sample mean/std on log-mels |

Training uses random cropping on longer recordings; validation and test use deterministic center crops. Multi-crop evaluation (3 or 5 evenly spaced crops with logit averaging) was tested as a separate experiment family.

For efficiency, recordings are decoded and resampled once into cached mono waveform tensors. The cache stores full waveforms, not fixed crops or spectrograms, so random cropping, augmentation, and log-mel extraction remain part of the training and evaluation pipeline.

### Implementation note

Some environments decode `.m4a` files through an external `ffmpeg` executable rather than through `torchaudio` alone. This does not change the feature pipeline; it only affects how raw audio is loaded.

## Model

The model is a compact 2D CNN over log-mel spectrograms. The convolutional layers learn local time-frequency patterns from singing audio, and adaptive average pooling converts the final feature map into a fixed-size representation for classification.

| Component | Details |
|---|---|
| Input shape | `[batch, 1, n_mels, time]` |
| Backbone | Conv2D blocks with BatchNorm, ReLU, MaxPool, Dropout |
| Pooling | Adaptive average pooling |
| Head | Linear classifier |
| Output | 3 age-bucket classes |

The architecture is intentionally small because the usable dataset contains only 1,197 recordings.

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Loss | Cross-entropy; optional class-weighted cross-entropy |
| Model selection metric | Validation macro-F1 |
| Reported metrics | Accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1 |
| Early stopping | Yes, on validation macro-F1 with patience |

Augmentation, when enabled, applies only to the training split:

- Waveform Gaussian noise after crop/pad and before log-mel extraction
- Time and frequency masking after log-mel extraction

Light and medium augmentation profiles were evaluated as part of the experiment sweep.

## Experiments

Experiments were organized as an ablation sequence:

| Experiment family | Purpose |
|---|---|
| Majority baseline | Establish a trivial reference point |
| Basic CNN | Test whether a scratch CNN learns useful signal from log-mels |
| Class-weighted CNN | Test whether class-weighted loss improves macro-F1 |
| Augmented CNN | Test whether light/medium augmentation improves generalization |
| Multi-crop evaluation | Test whether averaging deterministic crops improves recording-level stability |

Each family was run under a controlled sweep over learning rate, dropout, weight decay, and family-specific settings. The best run within each family was selected by validation macro-F1.

## Results

### Best run by family

| Family | Best run | Val macro-F1 | Test accuracy | Test macro-F1 | Test balanced accuracy |
|---|---|---:|---:|---:|---:|
| Majority baseline | `majority_baseline` | 0.1702 | 0.3277 | 0.1645 | 0.3333 |
| Basic CNN | `cnn_basic_lr0p001_drop0p35_wd0p0001` | 0.3904 | 0.4407 | 0.4079 | 0.4439 |
| Class-weighted CNN | `cnn_balanced_lr0p001_drop0p35_wd0p0001` | 0.3678 | 0.4350 | 0.3849 | 0.4392 |
| Light augmentation | `cnn_augmented_lr0p001_auglight` | 0.4180 | 0.4407 | 0.4225 | 0.4429 |
| Multi-crop evaluation | `cnn_augmented_multicrop_crop3` | 0.3043 | 0.3220 | 0.2950 | 0.3209 |

The final model was selected by validation macro-F1: **`cnn_augmented_lr0p001_auglight`**.

Final test metrics for the selected model:

| Metric | Value |
|---|---:|
| Accuracy | 0.4407 |
| Macro-F1 | 0.4225 |
| Balanced accuracy | 0.4429 |

Per-class test metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `under_25` | 0.4923 | 0.5424 | 0.5161 | 59 |
| `age_25_34` | 0.4800 | 0.2000 | 0.2824 | 60 |
| `age_35_plus` | 0.3908 | 0.5862 | 0.4690 | 58 |

### Interpretation

The scratch CNN substantially outperformed the majority baseline. The best basic CNN reached test macro-F1 **0.4079**. Light augmentation improved the validation-selected result to test macro-F1 **0.4225**. Class weighting alone did not improve over the best basic CNN. Medium augmentation did not help. Multi-crop evaluation was tested but did not improve performance.

The model performed best on the youngest and oldest buckets and struggled most with the middle `age_25_34` bucket, with recall **0.2000**. This is plausible because the middle bucket lies between two neighboring age groups and may share vocal characteristics with both. The results suggest that the model learned coarse age-related vocal cues, but adjacent adult age boundaries remain noisy. All three classes achieved nonzero F1 on the test set.

Performance is moderate overall; the system demonstrates meaningful signal but is not production-ready.

## Reproducing the Pipeline

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build if GPU training is desired. The code imports the package as `Sandbox.singerclassifier`, so set `PYTHONPATH` to the parent directory that contains the `Sandbox` package layout:

```bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
```

Generated artifacts such as split CSVs, cached waveforms, sweep configs, and experiment outputs should be stored outside the synced source tree if your deployment workflow overwrites the repository folder.

### 1. Inspect data (optional)

```bash
python -m Sandbox.singerclassifier.scripts.inspect_data \
  --data-root /path/to/DAMP-S-AG
```

### 2. Prepare splits

```bash
python -m Sandbox.singerclassifier.scripts.prepare_splits \
  --data-root /path/to/DAMP-S-AG \
  --output-csv /path/to/artifacts/processed/damp_sag_splits.csv \
  --summary-json /path/to/artifacts/data_inspection/split_summary.json \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

### 3. Precompute decoded-audio cache

```bash
python -m Sandbox.singerclassifier.scripts.precompute_audio_cache \
  --split-csv /path/to/artifacts/processed/damp_sag_splits.csv \
  --cache-dir /path/to/artifacts/cache/audio_22050_mono \
  --sample-rate 22050
```

### 4. Generate experiment configs

Update artifact paths in `configs/phase6_sweeps.yaml` if needed, then:

```bash
python -m Sandbox.singerclassifier.scripts.generate_sweep_configs \
  --sweep-spec configs/phase6_sweeps.yaml
```

### 5. Train and evaluate a run

Train a single generated config:

```bash
python -m Sandbox.singerclassifier.scripts.train \
  --config /path/to/artifacts/generated_configs/phase6/cnn_augmented_lr0p001_auglight.yaml
```

Or run one manifest entry (trains, then evaluates on the test split):

```bash
python -m Sandbox.singerclassifier.scripts.run_sweep_manifest \
  --manifest /path/to/artifacts/manifests/phase6_sweep_manifest.csv \
  --index 10
```

Evaluate a saved run:

```bash
python -m Sandbox.singerclassifier.scripts.evaluate \
  --run-dir /path/to/experiments/cnn_augmented_lr0p001_auglight \
  --split test
```

### 6. Summarize sweep results

```bash
python -m Sandbox.singerclassifier.scripts.summarize_sweep \
  --manifest /path/to/artifacts/manifests/phase6_sweep_manifest.csv \
  --output-root /path/to/experiments
```

## Repository Structure

```text
singerclassifier/
  audio.py                 # audio loading and waveform cache utilities
  data.py                  # dataset and dataloader
  features.py              # log-mel feature extraction and spectrogram augmentation
  models.py                # scratch CNN model
  metrics.py               # classification metrics and plots
  train_utils.py           # training, evaluation, and multi-crop helpers
  splits.py                # account-level split logic
  sweep.py                 # sweep config generation
  experiments.py           # experiment helpers

scripts/
  inspect_data.py
  prepare_splits.py
  precompute_audio_cache.py
  generate_sweep_configs.py
  run_sweep_manifest.py
  summarize_sweep.py
  train.py
  evaluate.py
  run_majority_baseline.py
  smoke_augmentation.py
  smoke_multicrop_eval.py

configs/
  phase6_sweeps.yaml
  majority_baseline.yaml
  cnn_basic.yaml
  cnn_balanced.yaml
  cnn_augmented.yaml
  cnn_augmented_multicrop.yaml
```

## Limitations

- The usable dataset is small: **1,197** labeled recordings.
- All recordings are of a single song: “Amazing Grace.”
- Age labels are derived from metadata and may be noisy.
- Metadata distributions for gender, device, and country are imbalanced.
- The middle age bucket (`age_25_34`) is the hardest to classify.
- Pretrained audio representations were not used, by design.

## Additional Notes

Detailed implementation history and milestone notes are recorded in `README_LOG.md`.
