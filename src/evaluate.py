"""Avaliação e exportação de métricas do modelo final.

Este módulo gera as evidências numéricas usadas no relatório: acurácia,
F1-score macro, classification report e matriz de confusão. Ele é usado no
treino opcional e nos notebooks para manter o mesmo cálculo em todas as etapas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from utils import save_json


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader, current_device: torch.device) -> tuple[list[int], list[int]]:
    """Coleta classes reais e preditas de um DataLoader sem gradientes.

    A saída é propositalmente simples: duas listas de inteiros. As funções do
    scikit-learn esperam esse formato para calcular as métricas.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in tqdm(loader, leave=False):
        # Labels do MedMNIST chegam no formato [batch, 1]; ``view(-1)`` converte
        # para [batch], que é o formato esperado pela CrossEntropy e pelas métricas.
        images = images.to(current_device, non_blocking=True)
        labels = labels.view(-1).long().to(current_device, non_blocking=True)
        logits = model(images)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    if not y_true:
        raise ValueError("loader produced no samples")
    return y_true, y_pred


def save_classification_artifacts(
    y_true: list[int],
    y_pred: list[int],
    class_names: tuple[str, ...],
    output_dir: str | Path,
    prefix: str = "test",
) -> dict[str, float]:
    """Salva métricas, classification report e matriz de confusão.

    Os arquivos gerados aqui são os que o professor pode conferir em
    ``experiments/final/`` para validar os números citados no relatório.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    save_json(output_dir / f"{prefix}_metrics.json", metrics)

    # ``zero_division=0`` evita erro caso alguma classe não seja predita em uma
    # execução ruim; o comportamento fica explícito e reproduzível.
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir / f"{prefix}_classification_report.csv", index=True)

    # A ordem das classes vem do MedMNIST e é reutilizada como linhas/colunas da
    # matriz, evitando confundir classes no relatório.
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output_dir / f"{prefix}_confusion_matrix.csv")
    np.save(output_dir / f"{prefix}_confusion_matrix.npy", matrix)
    return metrics
