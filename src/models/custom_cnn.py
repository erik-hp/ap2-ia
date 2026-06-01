"""CNN autoral de referencia para o PathMNIST 224x224."""

from __future__ import annotations

import torch
from torch import nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 9, dropout: float = 0.3):
        super().__init__()

        # Tres blocos Conv2d -> BatchNorm -> ReLU -> MaxPool, conforme o
        # requisito da Etapa 3.
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
        )

        # AdaptiveAvgPool remove a dependencia do tamanho espacial antes da
        # camada linear final.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        # bias=False porque o BatchNorm ja aprende deslocamento (beta).
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
