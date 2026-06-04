"""MLP configuravel com forward/backward explicitos em NumPy."""

from __future__ import annotations

import numpy as np

from .losses import cross_entropy_loss


class NumpyMLP:
    """MLP totalmente implementada com NumPy e backpropagation explicito.

    :param layer_sizes: Quantidade de neuronios da entrada ate a saida.
    :param seed: Seed usada na inicializacao He.
    :raises ValueError: Se a arquitetura for invalida.
    """

    def __init__(self, layer_sizes: list[int], seed: int = 42):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain input and output sizes")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("all layer sizes must be positive")

        rng = np.random.default_rng(seed)
        self.layer_sizes = [int(size) for size in layer_sizes]
        self.params: dict[str, np.ndarray] = {}

        # Inicializacao He: boa escolha para camadas com ReLU, pois preserva
        # melhor a escala dos sinais durante a propagacao.
        for i, (fan_in, fan_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:]), start=1):
            self.params[f"W{i}"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
            self.params[f"b{i}"] = np.zeros((1, fan_out))

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """Normaliza formatos comuns para ``[batch, features]``.

        Aceita amostras achatadas, imagens grayscale ``N x H x W`` e imagens
        RGB em ``N x H x W x 3`` ou ``N x 3 x H x W``. A funcao nao
        redimensiona imagens silenciosamente, pois isso esconderia erros de
        configuracao experimental.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("Input must contain at least one sample; received a scalar.")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[0] == 0:
            raise ValueError("Input batch must not be empty.")
        expected_features = self.layer_sizes[0]
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.mean(axis=-1)
        elif x.ndim == 4 and x.shape[1] == 3:
            x = x.mean(axis=1)
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)

        if x.shape[1] == expected_features:
            return x

        # O PathMNIST 28x28 pode chegar achatado como RGB: 28*28*3 = 2352.
        # Como a Etapa 1 pede entrada 784, convertemos para grayscale aqui.
        if x.shape[1] == expected_features * 3:
            return x.reshape(x.shape[0], expected_features, 3).mean(axis=2)

        raise ValueError(
            f"Input shape is incompatible after preprocessing: received {x.shape[1]} features per sample, "
            f"but the model expects {expected_features}. Accepted RGB flattened size: {expected_features * 3}. "
            "Check image dimensions, channel order, and layer_sizes[0]."
        )

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        """Executa forward e retorna logits e caches do backpropagation."""
        a = self._prepare_input(x)
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
        """Calcula gradientes dos parametros a partir do gradiente dos logits."""
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
        """Calcula CrossEntropy e gradientes para um batch."""
        # Funcao auxiliar usada no treino e no gradient checking.
        logits, caches = self.forward(x)
        loss, d_logits = cross_entropy_loss(logits, y)
        return loss, self.backward(d_logits, caches)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Retorna o indice da classe mais provavel por amostra."""
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=1)
