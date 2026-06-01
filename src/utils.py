"""Utilitarios compartilhados para experimentos reproduziveis."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch


RESULT_COLUMNS = [
    "modelo",
    "modo",
    "otimizador",
    "lr",
    "epoca",
    "loss_train",
    "loss_val",
    "acc_train",
    "acc_val",
    "tempo_s",
    "vram_mb",
]


def set_seed(seed: int = 42) -> None:
    # Mantem NumPy, Python e PyTorch sincronizados na mesma seed global.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Determinismo ajuda na reproducibilidade, embora possa reduzir desempenho.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_result(path: str | Path, row: dict[str, object]) -> None:
    # Cria o CSV com cabecalho na primeira escrita e apenas adiciona linhas depois.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        # Retorna True quando o treinamento deve parar.
        improved = self.best is None
        if self.best is not None and self.mode == "min":
            improved = value < self.best
        if self.best is not None and self.mode == "max":
            improved = value > self.best

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
