"""Testes dos pipelines e DataLoaders."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

import data.dataset as dataset_module


def test_build_transform_policies() -> None:
    for policy in ("none", "basic", "randaugment", "autoaugment"):
        transform = dataset_module.build_transform(image_size=224, train=True, augment_policy=policy)
        assert isinstance(transform, transforms.Compose)
        assert isinstance(transform.transforms[-2], transforms.ToTensor)
        assert isinstance(transform.transforms[-1], transforms.Normalize)


def test_get_dataset_uses_official_memmap_for_224(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(dataset_module, "MemoryMappedPathMNIST", lambda **kwargs: sentinel)
    result = dataset_module.get_dataset("train", source_size=224, download=False)
    assert result is sentinel


def test_get_dataset_uses_medmnist_for_smaller_source(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(dataset_module, "PathMNIST", lambda **kwargs: sentinel)
    result = dataset_module.get_dataset("train", source_size=28, download=False)
    assert result is sentinel


def test_get_loaders_configuration(monkeypatch) -> None:
    fake = TensorDataset(torch.zeros(8, 3, 224, 224), torch.zeros(8, 1, dtype=torch.long))
    monkeypatch.setattr(dataset_module, "get_dataset", lambda *args, **kwargs: fake)
    loaders = dataset_module.get_loaders(batch_size=4, source_size=224, num_workers=0, download=False)
    assert set(loaders) == {"train", "val", "test"}
    assert loaders["train"].batch_size == 4
    assert loaders["train"].num_workers == 0
    assert loaders["val"].batch_size == 4
