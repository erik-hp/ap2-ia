"""Fabrica de modelos para transfer learning com backbones do torchvision."""

from __future__ import annotations

import warnings

from torch import nn
from torchvision import models


MODEL_NAMES = ("resnet50", "efficientnet_b0", "mobilenet_v3_large", "vit_b_16")


def _set_trainable(model: nn.Module, trainable: bool) -> None:
    # Liga/desliga o calculo de gradientes de todos os parametros do backbone.
    for param in model.parameters():
        param.requires_grad = trainable


def _build_torchvision_model(factory, weights):
    try:
        return factory(weights=weights)
    except Exception as exc:
        if weights is None:
            raise
        warnings.warn(
            f"Could not load pretrained weights ({exc}). Falling back to random initialization.",
            RuntimeWarning,
            stacklevel=2,
        )
        return factory(weights=None)


def create_model(name: str, num_classes: int = 9, pretrained: bool = True, mode: str = "feature_extraction") -> nn.Module:
    """Cria backbone torchvision com classificador adaptado ao PathMNIST."""
    if mode not in {"feature_extraction", "fine_tuning"}:
        raise ValueError("mode must be 'feature_extraction' or 'fine_tuning'")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    name = name.lower()

    if name == "resnet50":
        # Cada arquitetura tem um nome diferente para o classificador final.
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = _build_torchvision_model(models.resnet50, weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        classifier = model.fc
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = _build_torchvision_model(models.efficientnet_b0, weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        classifier = model.classifier[-1]
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = _build_torchvision_model(models.mobilenet_v3_large, weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        classifier = model.classifier[-1]
    elif name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = _build_torchvision_model(models.vit_b_16, weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        classifier = model.heads.head
    else:
        raise ValueError(f"Unknown model: {name}. Available models: {', '.join(MODEL_NAMES)}")

    if mode == "feature_extraction":
        # No modo feature extraction, congela o backbone e treina apenas a
        # camada final substituida por Linear(..., 9).
        _set_trainable(model, False)
        for param in classifier.parameters():
            param.requires_grad = True
    return model


def parameter_groups(model: nn.Module, lr_base: float, mode: str) -> list[dict[str, object]]:
    """Separa backbone e classificador para learning rates diferentes."""
    if lr_base <= 0:
        raise ValueError("lr_base must be positive")
    if mode not in {"feature_extraction", "fine_tuning"}:
        raise ValueError("mode must be 'feature_extraction' or 'fine_tuning'")
    # Fine-tuning usa LR menor no backbone e LR base no classificador.
    classifier_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target = classifier_params if any(key in name for key in ("fc", "classifier", "heads.head")) else backbone_params
        target.append(param)

    if mode == "fine_tuning" and backbone_params:
        return [
            {"params": backbone_params, "lr": lr_base / 10.0},
            {"params": classifier_params, "lr": lr_base},
        ]
    return [{"params": classifier_params or [p for p in model.parameters() if p.requires_grad], "lr": lr_base}]
