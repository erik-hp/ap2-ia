"""Helpers para visualizar mapas de ativacao internos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def capture_first_conv_feature_maps(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Captura a saida da primeira convolucao usando forward hook temporario."""
    # Hook captura a saida da primeira camada convolucional sem alterar o modelo.
    first_conv = next((module for module in model.modules() if isinstance(module, torch.nn.Conv2d)), None)
    if first_conv is None:
        raise ValueError("Model has no Conv2d layer for feature-map visualization")
    captured = {}

    def hook(_, __, output):
        captured["maps"] = output.detach().cpu()

    handle = first_conv.register_forward_hook(hook)
    model.eval()
    try:
        with torch.no_grad():
            model(images)
    finally:
        handle.remove()
    if "maps" not in captured:
        raise RuntimeError("The forward hook did not capture feature maps")
    return captured["maps"]


def plot_16_feature_maps(feature_maps: torch.Tensor, image_index: int = 0, max_maps: int = 16):
    """Plota ate 16 mapas de ativacao em uma grade 4x4."""
    # O enunciado pede grid de 16 filtros e titulo com o indice de cada filtro.
    if feature_maps.ndim != 4:
        raise ValueError("feature_maps must have shape [batch, channels, height, width]")
    if image_index < 0:
        raise IndexError("image_index must be non-negative")
    if image_index >= feature_maps.shape[0]:
        raise IndexError("image_index is outside the captured batch")
    if max_maps <= 0:
        raise ValueError("max_maps must be positive")
    count = min(max_maps, feature_maps.shape[1])
    maps = feature_maps[image_index, :count]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for idx, ax in enumerate(axes.flat):
        if idx < count:
            ax.imshow(maps[idx], cmap="viridis")
            ax.set_title(str(idx))
        ax.axis("off")
    fig.tight_layout()
    return fig
