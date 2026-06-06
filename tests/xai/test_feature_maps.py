"""Testes de mapas de ativação."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from models.custom_cnn import CustomCNN
from xai.feature_maps import capture_first_conv_feature_maps, plot_16_feature_maps


def test_capture_first_conv_feature_maps() -> None:
    model = CustomCNN()
    maps = capture_first_conv_feature_maps(model, torch.randn(2, 3, 64, 64))
    assert maps.shape[0] == 2
    assert maps.shape[1] == 32


def test_plot_16_feature_maps() -> None:
    figure = plot_16_feature_maps(torch.randn(1, 16, 8, 8))
    assert len(figure.axes) == 16
    plt.close(figure)
