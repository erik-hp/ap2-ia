"""Testes da CNN autoral."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from models.custom_cnn import CustomCNN


def test_custom_cnn_forward_shape() -> None:
    model = CustomCNN(num_classes=5, dropout=0.2)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 5)


def test_custom_cnn_block_contract() -> None:
    block = CustomCNN._block(3, 16)
    assert isinstance(block, nn.Sequential)
    assert any(isinstance(layer, nn.Conv2d) for layer in block)
    assert any(isinstance(layer, nn.MaxPool2d) for layer in block)


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_custom_cnn_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError):
        CustomCNN(dropout=dropout)
