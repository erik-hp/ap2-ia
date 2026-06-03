"""Loaders do PathMNIST/MedMNIST usados pelos notebooks e treinos."""

from __future__ import annotations

import os
from typing import Literal

import torch
from medmnist import INFO, PathMNIST
from torch.utils.data import DataLoader
from torchvision import transforms


Split = Literal["train", "val", "test"]


PATHMNIST_CLASSES = tuple(label for _, label in sorted(INFO["pathmnist"]["label"].items(), key=lambda item: int(item[0])))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int = 224, train: bool = False, augment_policy: str = "basic") -> transforms.Compose:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if augment_policy not in {"none", "basic", "randaugment", "autoaugment"}:
        raise ValueError("augment_policy must be one of: none, basic, randaugment, autoaugment")
    steps: list[object] = []
    if image_size != 224:
        steps.append(transforms.Resize((image_size, image_size), antialias=True))
    if train:
        if augment_policy == "basic":
            steps.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)])
        elif augment_policy == "randaugment":
            steps.extend([transforms.RandAugment(num_ops=2, magnitude=7), transforms.RandomHorizontalFlip()])
        elif augment_policy == "autoaugment":
            steps.extend([transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET), transforms.RandomHorizontalFlip()])
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def get_dataset(
    split: Split,
    image_size: int = 224,
    train_transform: bool | None = None,
    download: bool = True,
    augment_policy: str = "basic",
) -> PathMNIST:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    if train_transform is None:
        train_transform = split == "train"
    return PathMNIST(
        split=split,
        transform=build_transform(image_size=image_size, train=train_transform, augment_policy=augment_policy),
        download=download,
        size=image_size,
        as_rgb=True,
    )


def _default_workers() -> int:
    if os.name == "nt":
        return 0
    return min(2, os.cpu_count() or 0)


def get_loaders(
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int | None = None,
    download: bool = True,
    augment_policy: str = "basic",
) -> dict[str, DataLoader]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    num_workers = _default_workers() if num_workers is None else num_workers
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    pin_memory = torch.cuda.is_available()
    datasets = {
        "train": get_dataset("train", image_size=image_size, train_transform=True, download=download, augment_policy=augment_policy),
        "val": get_dataset("val", image_size=image_size, train_transform=False, download=download, augment_policy="none"),
        "test": get_dataset("test", image_size=image_size, train_transform=False, download=download, augment_policy="none"),
    }
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        for split, dataset in datasets.items()
    }
