"""Helpers para visualizar mapas de ativacao internos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def capture_first_conv_feature_maps(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    # Hook captura a saida da primeira camada convolucional sem alterar o modelo.
    first_conv = next(module for module in model.modules() if isinstance(module, torch.nn.Conv2d))
    captured = {}

    def hook(_, __, output):
        captured["maps"] = output.detach().cpu()

    handle = first_conv.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(images)
    handle.remove()
    return captured["maps"]


def plot_16_feature_maps(feature_maps: torch.Tensor, image_index: int = 0):
    # O enunciado pede grid de 16 filtros e titulo com o indice de cada filtro.
    maps = feature_maps[image_index, :16]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(maps[idx], cmap="viridis")
        ax.set_title(str(idx))
        ax.axis("off")
    fig.tight_layout()
    return fig
