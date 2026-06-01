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
    target_layers = [last_conv_layer(model)]
    targets = None if target_class is None else [ClassifierOutputTarget(target_class)]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    return show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)
