"""Otimizadores implementados sem frameworks de deep learning."""

from __future__ import annotations

import numpy as np


class SGDMomentum:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-2, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for name, grad in grads.items():
            # Momentum acumula uma media exponencial do gradiente e suaviza
            # atualizacoes muito ruidosas.
            self.velocity[name] = self.beta * self.velocity[name] + self.lr * grad
            params[name] -= self.velocity[name]
