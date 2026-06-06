"""CNN autoral de referência para o PathMNIST 224x224.

Esta arquitetura atende à parte da Etapa 3 que pede uma CNN própria, treinada
do zero, para servir de base de comparação contra modelos pré-treinados.
"""

from __future__ import annotations

import torch
from torch import nn


class CustomCNN(nn.Module):
    """CNN autoral com tres blocos convolucionais, pooling e dropout.

    :param num_classes: Quantidade de classes da camada final.
    :param dropout: Probabilidade de dropout antes do classificador.
    """
    def __init__(self, num_classes: int = 9, dropout: float = 0.3):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        # Três blocos Conv2d -> BatchNorm -> ReLU -> MaxPool, conforme o
        # requisito da Etapa 3.
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
        )

        # AdaptiveAvgPool remove a dependência do tamanho espacial antes da
        # camada linear final.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        # bias=False porque o BatchNorm já aprende deslocamento (beta).
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna logits para um batch de imagens RGB."""
        return self.classifier(self.features(x))

    def _init_weights(self) -> None:
        """Inicializa pesos de forma adequada para ReLU e classificação."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming/He e indicada para camadas convolucionais seguidas de ReLU.
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm com escala 1 e bias 0 começa como transformação neutra.
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # A camada final começa com pesos pequenos para evitar logits
                # iniciais muito grandes.
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)
