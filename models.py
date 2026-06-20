"""Scratch CNN models for log-mel spectrogram age-bucket classification.

The model is intentionally compact because the provided dataset subset is limited.
The convolutional layers learn local time-frequency patterns from the log-mel
spectrogram, such as harmonic structure, spectral envelope, vibrato-related
texture, and timbral cues. Adaptive average pooling makes the model less
sensitive to exact song alignment and allows fixed-size classification from
variable-length spectrogram inputs. The model is trained from scratch and does
not use pretrained audio representations.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class SmallMelCNN(nn.Module):
    """Compact 2D CNN over log-mel spectrograms."""

    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 1,
        dropout: float = 0.25,
        classifier_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        self.features = nn.Sequential(
            self._conv_block(input_channels, 32, dropout),
            self._conv_block(32, 64, dropout),
            self._conv_block(64, 128, dropout),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape [batch, channels, n_mels, time], "
                f"got rank {x.ndim} with shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {x.shape[1]}"
            )

        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(
    model_name: str = "small_mel_cnn",
    num_classes: int = 3,
    **kwargs: Any,
) -> nn.Module:
    """Build a model by name."""
    name = model_name.lower()
    if name == "small_mel_cnn":
        return SmallMelCNN(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def describe_model(model: nn.Module) -> dict[str, int | str]:
    """Return a brief model summary."""
    return {
        "model_class": model.__class__.__name__,
        "total_parameters": count_parameters(model, trainable_only=False),
        "trainable_parameters": count_parameters(model, trainable_only=True),
    }
