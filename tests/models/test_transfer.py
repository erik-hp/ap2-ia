"""Testes da fabrica de transfer learning."""

from __future__ import annotations

import torch
from torch import nn

import models.transfer as transfer


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.fc = nn.Linear(4, 1000)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(4, 1000))


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.heads = nn.Module()
        self.heads.head = nn.Linear(4, 1000)


def _tiny_builder(factory, weights):
    name = factory.__name__
    if name == "resnet50":
        return TinyResNet()
    if name in {"efficientnet_b0", "mobilenet_v3_large"}:
        return TinyClassifier()
    return TinyViT()


def test_create_model_all_backbones_and_modes(monkeypatch) -> None:
    monkeypatch.setattr(transfer, "_build_torchvision_model", _tiny_builder)
    for name in transfer.MODEL_NAMES:
        feature_model = transfer.create_model(name, num_classes=9, pretrained=False, mode="feature_extraction")
        trainable = [param for param in feature_model.parameters() if param.requires_grad]
        assert trainable
        assert len(trainable) < len(list(feature_model.parameters()))

        fine_model = transfer.create_model(name, num_classes=9, pretrained=False, mode="fine_tuning")
        assert all(param.requires_grad for param in fine_model.parameters())


def test_parameter_groups_use_smaller_backbone_lr() -> None:
    model = TinyResNet()
    groups = transfer.parameter_groups(model, lr_base=1e-3, mode="fine_tuning")
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 1e-3
