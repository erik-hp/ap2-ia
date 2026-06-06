"""Otimizadores implementados sem frameworks de deep learning."""

from __future__ import annotations

import numpy as np


class SGDMomentum:
    """SGD com Momentum implementado sem frameworks de deep learning."""
    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-2, beta: float = 0.9):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.lr = lr
        self.beta = beta
        self.velocity = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """Atualiza os parâmetros in-place usando os gradientes informados."""
        for name, grad in grads.items():
            if name not in params:
                raise KeyError(f"Unknown parameter in grads: {name}")
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(params[name])
            if params[name].shape != grad.shape:
                raise ValueError(f"Gradient shape for {name} does not match parameter shape")
            # Momentum acumula uma média exponencial do gradiente e suaviza
            # atualizações muito ruidosas.
            self.velocity[name] = self.beta * self.velocity[name] + self.lr * grad
            params[name] -= self.velocity[name]
