"""Helpers de Grad-CAM usando a biblioteca pytorch-grad-cam."""

from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    # Grad-CAM costuma ser aplicado na ultima camada convolucional antes do GAP.
    convs = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    if not convs:
        raise ValueError("Model has no Conv2d layer for Grad-CAM")
    return convs[-1]


def gradcam_overlay(model, image_tensor: torch.Tensor, rgb_image: np.ndarray, target_class: int | None = None):
    # target_class=None faz a biblioteca usar a classe predita pelo modelo.
    model.eval()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must have shape [channels, height, width] or [batch, channels, height, width]")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape [height, width, 3]")
    try:
        model_device = next(model.parameters()).device
        image_tensor = image_tensor.to(model_device)
    except StopIteration:
        pass
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0
    rgb_image = np.clip(rgb_image, 0.0, 1.0)
    target_layers = [last_conv_layer(model)]
    targets = None if target_class is None else [ClassifierOutputTarget(target_class)]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    return show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)
