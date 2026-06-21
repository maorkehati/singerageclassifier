"""Feature extraction for raw audio waveforms."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T

LOG_EPS = 1e-6
NORM_EPS = 1e-6


class LogMelSpectrogram(nn.Module):
    """Convert waveforms to per-sample normalized log-mel spectrograms."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 80,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

    def _normalize(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.ndim == 3:
            mean = spec.mean(dim=(-2, -1), keepdim=True)
            std = spec.std(dim=(-2, -1), keepdim=True)
        elif spec.ndim == 4:
            mean = spec.mean(dim=(-3, -2, -1), keepdim=True)
            std = spec.std(dim=(-3, -2, -1), keepdim=True)
        else:
            raise ValueError(
                f"Expected 3D or 4D spectrogram tensor, got shape {tuple(spec.shape)}"
            )
        return (spec - mean) / (std + NORM_EPS)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel(waveform)
        log_mel = torch.log(mel + LOG_EPS)
        if self.normalize:
            log_mel = self._normalize(log_mel)
        return log_mel


def assert_waveform_feature_extractor_same_device(
    waveforms: torch.Tensor,
    feature_extractor: nn.Module,
) -> None:
    """Raise if waveform and torchaudio STFT buffers are on different devices."""
    target_device = waveforms.device
    for buffer in feature_extractor.buffers():
        if buffer.device != target_device:
            raise RuntimeError(
                "STFT device mismatch: waveforms on "
                f"{target_device} but feature extractor buffer on {buffer.device}. "
                "Move waveform crops and LogMelSpectrogram to the same device before "
                "feature extraction."
            )
        return

    for param in feature_extractor.parameters():
        if param.device != target_device:
            raise RuntimeError(
                "Feature extractor parameter device mismatch: waveforms on "
                f"{target_device} but parameter on {param.device}."
            )
        return


class SpectrogramAugment(nn.Module):
    """Lightweight time/frequency masking for log-mel spectrograms."""

    def __init__(
        self,
        time_mask_cfg: dict | None = None,
        freq_mask_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        time_mask_cfg = time_mask_cfg or {}
        freq_mask_cfg = freq_mask_cfg or {}
        self.time_mask_enabled = bool(time_mask_cfg.get("enabled", False))
        self.freq_mask_enabled = bool(freq_mask_cfg.get("enabled", False))
        self.time_max_width = int(time_mask_cfg.get("max_width", 40))
        self.time_num_masks = int(time_mask_cfg.get("num_masks", 1))
        self.freq_max_width = int(freq_mask_cfg.get("max_width", 8))
        self.freq_num_masks = int(freq_mask_cfg.get("num_masks", 1))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        out = spec.clone()
        if out.ndim == 3:
            out = self._augment_single(out)
        elif out.ndim == 4:
            for idx in range(out.shape[0]):
                out[idx] = self._augment_single(out[idx])
        else:
            raise ValueError(f"Unexpected spectrogram shape: {tuple(out.shape)}")
        return out

    def _augment_single(self, spec: torch.Tensor) -> torch.Tensor:
        out = spec.clone()
        n_mels = out.shape[-2]
        n_time = out.shape[-1]

        if self.freq_mask_enabled and n_mels > 1:
            for _ in range(self.freq_num_masks):
                width = min(
                    self.freq_max_width,
                    max(1, torch.randint(1, n_mels, (1,)).item()),
                )
                start = torch.randint(0, max(n_mels - width + 1, 1), (1,)).item()
                out[..., start : start + width, :] = 0.0

        if self.time_mask_enabled and n_time > 1:
            for _ in range(self.time_num_masks):
                width = min(
                    self.time_max_width,
                    max(1, torch.randint(1, n_time, (1,)).item()),
                )
                start = torch.randint(0, max(n_time - width + 1, 1), (1,)).item()
                out[..., :, start : start + width] = 0.0

        return out
