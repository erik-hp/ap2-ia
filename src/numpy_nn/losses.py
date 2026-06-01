"""Funcoes de loss para a MLP implementada apenas com NumPy."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    # Subtrair o maior logit evita overflow em exp(logits).
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Retorna CrossEntropy estavel e o gradiente em relacao aos logits."""
    y = y.reshape(-1).astype(int)
    probs = softmax(logits)
    n = logits.shape[0]
    loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))

    # Para softmax + cross-entropy, o gradiente simplifica para:
    # probs - one_hot(y), normalizado pelo tamanho do batch.
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return float(loss), grad
