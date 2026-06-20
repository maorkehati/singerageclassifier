"""Audio loading and fixed-length waveform preparation."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio

CACHE_VERSION = "audio_22050_mono_v1"
DEFAULT_CACHE_SAMPLE_RATE = 22050


def _safe_cache_stem(performance_id: str, source_path: Path | None = None) -> str:
    stem = str(performance_id).strip()
    if stem:
        return re.sub(r"[^\w\-+.]", "_", stem)
    if source_path is None:
        raise ValueError("performance_id is required when source_path is not provided")
    digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()
    return digest[:16]


def audio_cache_path(
    cache_dir: str | Path,
    performance_id: str,
    source_path: str | Path | None = None,
) -> Path:
    """Return the deterministic cache file path for a performance."""
    cache_root = Path(cache_dir)
    stem = _safe_cache_stem(performance_id, Path(source_path) if source_path else None)
    return cache_root / f"{stem}.pt"


def validate_cached_waveform_payload(
    payload: dict,
    expected_sample_rate: int = DEFAULT_CACHE_SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    if not isinstance(payload, dict):
        raise ValueError(f"Cache payload must be a dict, got {type(payload).__name__}")

    cache_version = payload.get("cache_version")
    if cache_version != CACHE_VERSION:
        raise ValueError(
            f"Unsupported cache version: {cache_version!r} (expected {CACHE_VERSION})"
        )

    if "waveform" not in payload:
        raise ValueError("Cache payload missing 'waveform'")

    waveform = payload["waveform"]
    if not isinstance(waveform, torch.Tensor):
        raise ValueError(f"Cache waveform must be a torch.Tensor, got {type(waveform).__name__}")

    waveform = waveform.to(torch.float32)
    if waveform.ndim != 2:
        raise ValueError(
            f"Cache waveform must have shape [channels, samples], got {tuple(waveform.shape)}"
        )
    if waveform.numel() == 0:
        raise ValueError("Cache waveform is empty")
    if not torch.isfinite(waveform).all():
        raise ValueError("Cache waveform contains NaN or Inf values")

    sample_rate = payload.get("sample_rate")
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Cache sample rate {sample_rate!r} != expected {expected_sample_rate}"
        )

    return waveform, int(sample_rate)


def load_cached_waveform(
    cache_path: str | Path,
    expected_sample_rate: int = DEFAULT_CACHE_SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """Load and validate a cached waveform tensor."""
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio cache not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return validate_cached_waveform_payload(payload, expected_sample_rate)


def save_waveform_cache(
    cache_path: str | Path,
    waveform: torch.Tensor,
    sample_rate: int,
    source_path: str | Path,
) -> None:
    """Save a decoded waveform to the persistent cache."""
    source = Path(source_path)
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "waveform": waveform.to(torch.float32).cpu(),
        "sample_rate": int(sample_rate),
        "source_path": str(source),
        "source_size": int(source.stat().st_size),
        "source_mtime": float(source.stat().st_mtime),
        "cache_version": CACHE_VERSION,
    }
    torch.save(payload, path)


def _load_audio_ffmpeg(
    path: Path,
    target_sample_rate: int,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Decode audio using the ffmpeg executable and return a float waveform tensor."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg executable not found in PATH. "
            "Install ffmpeg or ensure it is available on the system."
        )

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1" if mono else "1",
        "-ar",
        str(target_sample_rate),
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed to decode {path} (exit {proc.returncode}): {stderr or 'no stderr'}"
        )

    if not proc.stdout:
        raise RuntimeError(f"ffmpeg decoded empty output for {path}")

    audio_np = np.frombuffer(proc.stdout, dtype=np.float32)
    waveform = torch.from_numpy(audio_np.copy()).unsqueeze(0)
    return waveform, target_sample_rate


def _load_audio_torchaudio(
    path: Path,
    target_sample_rate: int,
    mono: bool,
) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.to(torch.float32)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    return waveform, sample_rate


def load_audio(
    path: str | Path,
    target_sample_rate: int = 22050,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load an audio file and return a float waveform tensor."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    suffix = audio_path.suffix.lower()
    errors: list[str] = []

    if suffix == ".m4a":
        try:
            return _load_audio_ffmpeg(audio_path, target_sample_rate, mono)
        except Exception as exc:
            errors.append(f"ffmpeg: {exc}")
    else:
        try:
            return _load_audio_torchaudio(audio_path, target_sample_rate, mono)
        except Exception as exc:
            errors.append(f"torchaudio: {exc}")

        try:
            return _load_audio_ffmpeg(audio_path, target_sample_rate, mono)
        except Exception as exc:
            errors.append(f"ffmpeg: {exc}")

    attempts = "\n".join(f"  - {entry}" for entry in errors)
    raise RuntimeError(
        f"Failed to load audio: {audio_path}\nAttempted backends:\n{attempts}"
    )


def crop_or_pad_waveform(
    waveform: torch.Tensor,
    target_num_samples: int,
    random_crop: bool = False,
) -> torch.Tensor:
    """Crop or zero-pad waveform to a fixed number of samples."""
    if waveform.ndim != 2:
        raise ValueError(
            f"Expected waveform shape [channels, num_samples], got {tuple(waveform.shape)}"
        )

    num_samples = waveform.shape[-1]
    if num_samples > target_num_samples:
        if random_crop:
            start = torch.randint(
                0, num_samples - target_num_samples + 1, (1,)
            ).item()
        else:
            start = (num_samples - target_num_samples) // 2
        return waveform[..., start : start + target_num_samples]

    if num_samples < target_num_samples:
        pad = target_num_samples - num_samples
        return torch.nn.functional.pad(waveform, (0, pad))

    return waveform


def crop_from_start(
    waveform: torch.Tensor,
    start: int,
    target_num_samples: int,
) -> torch.Tensor:
    """Extract a fixed-length crop starting at `start`, with zero-padding if needed."""
    segment = waveform[..., start : start + target_num_samples]
    if segment.shape[-1] < target_num_samples:
        pad = target_num_samples - segment.shape[-1]
        segment = torch.nn.functional.pad(segment, (0, pad))
    return segment


def deterministic_crop_starts(
    num_samples: int,
    target_num_samples: int,
    num_crops: int,
) -> list[int]:
    """Return deterministic crop start indices spaced across the waveform."""
    if num_crops <= 1:
        if num_samples > target_num_samples:
            return [(num_samples - target_num_samples) // 2]
        return [0]

    if num_samples <= target_num_samples:
        return [0] * num_crops

    max_start = num_samples - target_num_samples
    return [
        int(round(i * max_start / (num_crops - 1)))
        for i in range(num_crops)
    ]
