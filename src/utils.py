"""Utilitários compartilhados para experimentos reproduzíveis.

Funções daqui aparecem em várias etapas: seed global, escolha de dispositivo,
registro de métricas em CSV/JSON, hardware e early stopping. Centralizar esses
blocos evita divergência entre notebooks e scripts.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch


RESULT_COLUMNS = [
    "tag",
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
# Cabeçalho único dos CSVs de experimentos. Manter a ordem fixa facilita
# comparar execuções e importar os resultados no relatório.


def _lr_tag(value: object) -> str:
    """Formata learning rate no mesmo padrão dos manifests."""
    try:
        lr = float(value)
    except (TypeError, ValueError):
        return str(value).replace(".", "p")
    if abs(lr - 1e-2) < 1e-12:
        return "1e2"
    if abs(lr - 1e-3) < 1e-12:
        return "1e3"
    if abs(lr - 1e-4) < 1e-12:
        return "1e4"
    return f"{lr:g}".replace(".", "p").replace("-", "")


def _infer_result_tag(row: dict[str, object]) -> str:
    """Infere tag para CSVs antigos que não tinham essa coluna."""
    if row.get("tag"):
        return str(row["tag"])
    model = str(row.get("modelo", "model"))
    mode = str(row.get("modo", "feature_extraction"))
    optimizer = str(row.get("otimizador", "opt"))
    lr = _lr_tag(row.get("lr", "lr"))
    if model == "custom_cnn":
        return f"{model}_{optimizer}_{lr}"
    return f"{model}_{mode}_{optimizer}_{lr}"


def _migrate_result_csv(path: Path) -> bool:
    """Atualiza CSV antigo para o cabeçalho atual sem perder linhas."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames == RESULT_COLUMNS:
            return True
        rows = list(reader)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            row["tag"] = _infer_result_tag(row)
            writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})
    return True


def project_root() -> Path:
    """Retorna a raiz do repositório a partir de ``src/utils.py``."""
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    """Sincroniza seeds de Python, NumPy e PyTorch."""
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
    """Seleciona CUDA quando disponível; caso contrário, usa CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_result(path: str | Path, row: dict[str, object]) -> None:
    """Adiciona uma linha padronizada ao CSV de experimentos."""
    # Cria o CSV com cabeçalho na primeira escrita e apenas adiciona linhas depois.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = _migrate_result_csv(path)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def save_json(path: str | Path, data: dict[str, object]) -> None:
    """Salva dicionário como JSON UTF-8 indentado."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def collect_hardware_info() -> dict[str, object]:
    """Coleta informações reproduzíveis do ambiente e do acelerador."""
    info: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": props.name,
                "vram_total_mb": round(props.total_memory / 1024**2, 2),
            }
        )
    return info


class Timer:
    """Context manager para medir tempo decorrido.

    Usado por época em ``train.py`` para registrar o custo temporal de cada
    arquitetura na comparação.
    """
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


class EarlyStopping:
    """Controla parada antecipada baseada em métrica de validação.

    Evita treinar muitas épocas sem melhora e registra uma estratégia comum de
    regularização/controle experimental para a Etapa 5.
    """
    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 0.0):
        if patience <= 0:
            raise ValueError("patience must be positive")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Atualiza o estado e informa se o treinamento deve parar."""
        # Retorna True quando o treinamento deve parar.
        improved = self.best is None
        if self.best is not None and self.mode == "min":
            improved = value < self.best - self.min_delta
        if self.best is not None and self.mode == "max":
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
