"""Testes dos helpers de Grad-CAM."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from xai.gradcam import gradcam_overlay, last_conv_layer


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))

    def forward(self, x):
        return self.head(self.features(x))


def test_last_conv_layer() -> None:
    model = TinyCNN()
    assert last_conv_layer(model) is model.features[2]


def test_gradcam_overlay_shape() -> None:
    overlay = gradcam_overlay(TinyCNN(), torch.randn(1, 3, 32, 32), np.zeros((32, 32, 3), dtype=np.float32), target_class=0)
    assert overlay.shape == (32, 32, 3)
