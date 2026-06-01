"""MLP configuravel com forward/backward explicitos em NumPy."""

from __future__ import annotations

import numpy as np

from .losses import cross_entropy_loss


class NumpyMLP:
    def __init__(self, layer_sizes: list[int], seed: int = 42):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain input and output sizes")

        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.params: dict[str, np.ndarray] = {}

        # Inicializacao He: boa escolha para camadas com ReLU, pois preserva
        # melhor a escala dos sinais durante a propagacao.
        for i, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), start=1):
            self.params[f"W{i}"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            self.params[f"b{i}"] = np.zeros((1, fan_out))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        a = x
        caches: list[dict[str, np.ndarray]] = []
        n_layers = len(self.layer_sizes) - 1

        for i in range(1, n_layers + 1):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache = {"a_prev": a, "z": z}

            # As camadas escondidas usam ReLU; a ultima camada retorna logits
            # crus, pois o softmax e aplicado dentro da loss numericamente estavel.
            if i < n_layers:
                a = np.maximum(0.0, z)
            else:
                a = z
            cache["a"] = a
            caches.append(cache)
        return a, caches

    def backward(self, d_logits: np.ndarray, caches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        grads: dict[str, np.ndarray] = {}
        da = d_logits

        # Backpropagation da ultima camada ate a primeira, aplicando a regra da
        # cadeia em forma matricial.
        for layer_index in range(len(caches), 0, -1):
            cache = caches[layer_index - 1]
            if layer_index < len(caches):
                dz = da * (cache["z"] > 0.0)
            else:
                dz = da

            # dW = A_anterior^T @ dZ e db = soma de dZ no batch.
            grads[f"W{layer_index}"] = cache["a_prev"].T @ dz
            grads[f"b{layer_index}"] = np.sum(dz, axis=0, keepdims=True)
            da = dz @ self.params[f"W{layer_index}"].T
        return grads

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        # Funcao auxiliar usada no treino e no gradient checking.
        logits, caches = self.forward(x)
        loss, d_logits = cross_entropy_loss(logits, y)
        return loss, self.backward(d_logits, caches)

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=1)
