"""Gradient checking por diferencas finitas para a MLP NumPy."""

from __future__ import annotations

import numpy as np

from .model import NumpyMLP


def gradient_check(model: NumpyMLP, x: np.ndarray, y: np.ndarray, eps: float = 1e-5, tolerance: float = 1e-5) -> float:
    if eps <= 0:
        raise ValueError("eps must be positive")
    loss, grads = model.loss_and_grads(x, y)
    del loss
    max_diff = 0.0

    # Cada peso e bias e perturbado individualmente para comparar o gradiente
    # analitico do backprop com o gradiente numerico.
    for name, param in model.params.items():
        grad = grads[name]
        for index in np.ndindex(param.shape):
            original = param[index]

            param[index] = original + eps
            loss_plus, _ = model.loss_and_grads(x, y)

            param[index] = original - eps
            loss_minus, _ = model.loss_and_grads(x, y)

            param[index] = original
            grad_num = (loss_plus - loss_minus) / (2.0 * eps)
            diff = abs(grad[index] - grad_num) / (abs(grad[index]) + abs(grad_num) + 1e-8)
            max_diff = max(max_diff, float(diff))

    # O limite 1e-5 e o criterio exigido no enunciado.
    assert max_diff < tolerance, f"Gradient check falhou: {max_diff}"
    return max_diff
