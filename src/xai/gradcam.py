"""Helpers de Grad-CAM usando a biblioteca pytorch-grad-cam.

Grad-CAM é a principal técnica de interpretabilidade visual da Etapa 4. O
resultado é um mapa de calor sobre a imagem original, indicando as regiões que
mais influenciaram a classe predita ou uma classe alvo escolhida.
"""

from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Retorna a última camada convolucional adequada ao Grad-CAM."""
    # Grad-CAM costuma ser aplicado na última camada convolucional antes do
    # pooling/classificador, pois ela combina informação semântica com alguma
    # localização espacial.
    convs = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    if not convs:
        raise ValueError("Model has no Conv2d layer for Grad-CAM")
    return convs[-1]


def gradcam_overlay(model, image_tensor: torch.Tensor, rgb_image: np.ndarray, target_class: int | None = None):
    """Gera overlay Grad-CAM RGB para uma imagem e classe alvo opcional.

    ``image_tensor`` e a imagem normalizada que entra no modelo. ``rgb_image`` e
    a mesma imagem em RGB [0, 1] para visualização no relatório.
    """
    # target_class=None faz a biblioteca usar a classe predita pelo modelo.
    model.eval()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must have shape [channels, height, width] or [batch, channels, height, width]")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape [height, width, 3]")
    try:
        # Garante que a imagem esteja no mesmo dispositivo do modelo, seja CPU
        # ou GPU, evitando erro durante o forward usado pelo Grad-CAM.
        model_device = next(model.parameters()).device
        image_tensor = image_tensor.to(model_device)
    except StopIteration:
        pass
    # A biblioteca espera imagem visual em float [0, 1], não normalizada por
    # ImageNet. Por isso normalizamos/clampamos antes do overlay.
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0
    rgb_image = np.clip(rgb_image, 0.0, 1.0)
    target_layers = [last_conv_layer(model)]
    targets = None if target_class is None else [ClassifierOutputTarget(target_class)]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    return show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)
