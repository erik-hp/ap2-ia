"""Pipelines oficiais do PathMNIST com suporte eficiente ao tamanho 224x224."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from medmnist import INFO, PathMNIST
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url


Split = Literal["train", "val", "test"]
AugmentPolicy = Literal["none", "basic", "randaugment", "autoaugment"]

PATHMNIST_CLASSES = tuple(label for _, label in sorted(INFO["pathmnist"]["label"].items(), key=lambda item: int(item[0])))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int = 224, train: bool = False, augment_policy: AugmentPolicy = "basic") -> transforms.Compose:
    """Cria o pipeline torchvision usado nas etapas PyTorch.

    :param image_size: Altura e largura finais entregues ao modelo.
    :param train: Ativa transformacoes estocasticas de treino.
    :param augment_policy: Politica de aumento aplicada somente ao treino.
    :return: Composicao de transformacoes com normalizacao ImageNet.
    :raises ValueError: Se ``image_size`` ou ``augment_policy`` forem invalidos.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if augment_policy not in {"none", "basic", "randaugment", "autoaugment"}:
        raise ValueError("augment_policy must be one of: none, basic, randaugment, autoaugment")

    steps: list[object] = []
    if train:
        if augment_policy == "basic":
            steps.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)])
        elif augment_policy == "randaugment":
            steps.extend([transforms.RandAugment(num_ops=2, magnitude=7), transforms.RandomHorizontalFlip()])
        elif augment_policy == "autoaugment":
            steps.extend([transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET), transforms.RandomHorizontalFlip()])
    steps.extend(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


class MemoryMappedPathMNIST(Dataset):
    """PathMNIST+ oficial extraido em ``.npy`` e acessado sem carregar tudo na RAM.

    O arquivo oficial ``pathmnist_224.npz`` e um ZIP de arrays NumPy. Extrair
    esses arrays uma vez permite usar ``mmap_mode='r'`` e carregar apenas as
    imagens solicitadas pelo DataLoader, preservando os splits oficiais.

    :param split: Split oficial: ``train``, ``val`` ou ``test``.
    :param root: Diretorio de cache do dataset.
    :param transform: Transformacao aplicada a cada imagem PIL.
    :param download: Baixa e extrai o arquivo oficial quando necessario.
    :raises FileNotFoundError: Se os dados nao existirem e ``download=False``.
    """

    def __init__(
        self,
        split: Split,
        root: str | Path | None = None,
        transform=None,
        download: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        self.split = split
        self.root = Path(root or Path.home() / ".medmnist")
        self.transform = transform
        self.extracted_dir = self.root / "pathmnist_224_memmap"
        if download:
            self._prepare()

        image_path = self.extracted_dir / f"{split}_images.npy"
        label_path = self.extracted_dir / f"{split}_labels.npy"
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(
                f"PathMNIST 224 files were not found in {self.extracted_dir}. "
                "Use download=True to download and extract the official dataset."
            )
        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")

    def _prepare(self) -> None:
        expected = [self.extracted_dir / f"{split}_{kind}.npy" for split in ("train", "val", "test") for kind in ("images", "labels")]
        if all(path.exists() for path in expected):
            return

        self.root.mkdir(parents=True, exist_ok=True)
        archive_name = "pathmnist_224.npz"
        archive_path = self.root / archive_name
        info = INFO["pathmnist"]
        # download_url valida o MD5 oficial e evita aceitar download parcial.
        download_url(info["url_224"], str(self.root), filename=archive_name, md5=info["MD5_224"])
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(self.extracted_dir)

    def __len__(self) -> int:
        """Retorna o numero de amostras do split."""
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor | Image.Image, np.ndarray]:
        """Carrega uma amostra individual do array mapeado em disco."""
        image = Image.fromarray(np.asarray(self.images[index]))
        if self.transform is not None:
            image = self.transform(image)
        # Labels sao pequenos; a copia evita avisos do PyTorch sobre arrays mmap
        # somente leitura durante o collate.
        return image, np.asarray(self.labels[index]).copy()


def get_dataset(
    split: Split,
    image_size: int = 224,
    source_size: int = 224,
    train_transform: bool | None = None,
    download: bool = True,
    augment_policy: AugmentPolicy = "basic",
    root: str | Path | None = None,
) -> Dataset:
    """Cria um split oficial do PathMNIST.

    :param split: Split oficial solicitado.
    :param image_size: Tamanho final entregue ao modelo.
    :param source_size: Versao oficial do MedMNIST. Nas etapas 2-5 deve ser 224.
    :param train_transform: Sobrescreve a ativacao de augmentations.
    :param download: Baixa o dataset quando ausente.
    :param augment_policy: Politica de aumento do split de treino.
    :param root: Diretorio opcional de cache.
    :return: Dataset oficial PathMNIST.
    :raises ValueError: Se ``split`` ou ``source_size`` forem invalidos.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    if source_size not in {28, 64, 128, 224}:
        raise ValueError("source_size must be one of: 28, 64, 128, 224")
    if train_transform is None:
        train_transform = split == "train"
    transform = build_transform(image_size=image_size, train=train_transform, augment_policy=augment_policy)
    if source_size == 224:
        return MemoryMappedPathMNIST(split=split, root=root, transform=transform, download=download)
    kwargs = {"split": split, "transform": transform, "download": download, "size": source_size, "as_rgb": True}
    if root is not None:
        kwargs["root"] = str(root)
    return PathMNIST(**kwargs)


def _default_workers() -> int:
    """Escolhe um numero conservador de workers para Windows e Colab."""
    if os.name == "nt":
        return 0
    return min(2, os.cpu_count() or 0)


def get_loaders(
    batch_size: int = 32,
    image_size: int = 224,
    source_size: int = 224,
    num_workers: int | None = None,
    download: bool = True,
    augment_policy: AugmentPolicy = "basic",
    root: str | Path | None = None,
) -> dict[str, DataLoader]:
    """Cria DataLoaders para os splits oficiais sem remixar dados.

    :param batch_size: Numero de amostras por batch.
    :param image_size: Tamanho final entregue ao modelo.
    :param source_size: Versao oficial do PathMNIST.
    :param num_workers: Processos do DataLoader; ``None`` usa valor conservador.
    :param download: Baixa os dados quando ausentes.
    :param augment_policy: Politica aplicada apenas ao treino.
    :param root: Diretorio opcional de cache.
    :return: Dicionario com loaders ``train``, ``val`` e ``test``.
    :raises ValueError: Se os parametros numericos forem invalidos.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if source_size not in {28, 64, 128, 224}:
        raise ValueError("source_size must be one of: 28, 64, 128, 224")
    num_workers = _default_workers() if num_workers is None else num_workers
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    pin_memory = torch.cuda.is_available()
    datasets = {
        "train": get_dataset("train", image_size, source_size, True, download, augment_policy, root),
        "val": get_dataset("val", image_size, source_size, False, download, "none", root),
        "test": get_dataset("test", image_size, source_size, False, download, "none", root),
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
