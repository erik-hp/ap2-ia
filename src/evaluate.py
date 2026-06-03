"""Avaliacao e exportacao de metricas do modelo final."""

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
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in tqdm(loader, leave=False):
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    save_json(output_dir / f"{prefix}_metrics.json", metrics)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir / f"{prefix}_classification_report.csv", index=True)

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output_dir / f"{prefix}_confusion_matrix.csv")
    np.save(output_dir / f"{prefix}_confusion_matrix.npy", matrix)
    return metrics
