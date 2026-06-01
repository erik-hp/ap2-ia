"""Fabrica de modelos para transfer learning com backbones do torchvision."""

from __future__ import annotations

from torch import nn
from torchvision import models


def _set_trainable(model: nn.Module, trainable: bool) -> None:
    # Liga/desliga o calculo de gradientes de todos os parametros do backbone.
    for param in model.parameters():
        param.requires_grad = trainable


def create_model(name: str, num_classes: int = 9, pretrained: bool = True, mode: str = "feature_extraction") -> nn.Module:
    if mode not in {"feature_extraction", "fine_tuning"}:
        raise ValueError("mode must be 'feature_extraction' or 'fine_tuning'")

    name = name.lower()

    if name == "resnet50":
        # Cada arquitetura tem um nome diferente para o classificador final.
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        classifier = model.fc
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        classifier = model.classifier[-1]
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        classifier = model.classifier[-1]
    elif name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        classifier = model.heads.head
    else:
        raise ValueError(f"Unknown model: {name}")

    if mode == "feature_extraction":
        # No modo feature extraction, congela o backbone e treina apenas a
        # camada final substituida por Linear(..., 9).
        _set_trainable(model, False)
        for param in classifier.parameters():
            param.requires_grad = True
    return model


def parameter_groups(model: nn.Module, lr_base: float, mode: str) -> list[dict[str, object]]:
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
