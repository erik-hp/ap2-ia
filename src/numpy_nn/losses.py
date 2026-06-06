"""Funções de loss para a MLP implementada apenas com NumPy."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Calcula softmax numericamente estável para logits 2D."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    # Subtrair o maior logit evita overflow em exp(logits).
    logits = logits.astype(np.float64, copy=False)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Retorna CrossEntropy estável e o gradiente em relação aos logits.

    :param logits: Matriz ``[batch, classes]``.
    :param y: Índices inteiros das classes.
    :return: Tupla ``(loss_média, gradiente_dos_logits)``.
    :raises ValueError: Se shapes ou índices forem inválidos.
    """
    y = y.reshape(-1).astype(int)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if logits.shape[0] == 0:
        raise ValueError("batch must not be empty")
    if y.shape[0] != logits.shape[0]:
        raise ValueError("y must have the same batch size as logits")
    if np.any((y < 0) | (y >= logits.shape[1])):
        raise ValueError("y contains class indices outside the logits range")

    probs = softmax(logits)
    n = logits.shape[0]
    loss = -np.mean(np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)))

    # Para softmax + cross-entropy, o gradiente simplifica para:
    # probs - one_hot(y), normalizado pelo tamanho do batch.
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return float(loss), grad
