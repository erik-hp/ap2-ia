# Hardware Utilizado

Este documento registra o ambiente computacional usado nas etapas do projeto e
serve como evidência complementar para o relatório.

## Ambiente

As execuções foram realizadas em ambientes de notebook em nuvem, principalmente
Google Colab e Kaggle Notebooks, conforme disponibilidade de GPU. Como esses
ambientes gratuitos podem variar entre sessões, CPU e RAM são descritos pelo
perfil do ambiente utilizado, e os valores de GPU/VRAM são documentados a partir
das execuções registradas.

| Recurso | Descrição documentada |
|---|---|
| CPU | vCPU x86_64 fornecida pelo ambiente Google Colab/Kaggle. |
| RAM | Memória RAM de sessão gratuita em notebook de nuvem, aproximadamente entre 12 GB e 16 GB conforme disponibilidade do ambiente. |
| GPU | NVIDIA Tesla T4 nas etapas com PyTorch. |
| VRAM | Aproximadamente 16 GB de VRAM total na Tesla T4. |

## Uso por etapa

| Etapa | Ambiente utilizado | Observação |
|---|---|---|
| 1 - MLP NumPy | Notebook em nuvem/CPU | Etapa leve, usando PathMNIST 28x28 para a implementação manual da MLP. |
| 2 - Validação PyTorch | Notebook em nuvem | Reimplementação equivalente da MLP em PyTorch para comparação com NumPy. |
| 3 - Comparação de arquiteturas | Google Colab/Kaggle com NVIDIA Tesla T4 | Execuções curtas para comparar CNN própria, ResNet50, EfficientNet-B0, MobileNetV3 Large e ViT-B/16. |
| 4 - XAI | Notebook em nuvem com PyTorch | Geração de Feature Maps e Grad-CAM a partir do modelo final. |
| 5 - Treinamento final | Google Colab/Kaggle com NVIDIA Tesla T4 | Treinamento final da ResNet50 em fine-tuning e avaliação única no teste. |

## Evidências registradas

Os logs salvos pelo projeto registram tempo de execução e pico de uso de VRAM:

- `experiments/stage03_best_by_run.csv`
- `experiments/final/final_training_history.csv`
- `experiments/hardware.json`

Na comparação de arquiteturas, o maior pico de VRAM registrado foi de
aproximadamente `3284,89 MB`, observado no ViT-B/16 em fine-tuning. No
treinamento final, o maior pico registrado foi de aproximadamente `1653,80 MB`.

## Observação metodológica

As limitações de tempo de GPU nos ambientes gratuitos influenciaram o número de
épocas e a profundidade da busca por hiperparâmetros. Por isso, a Etapa 3 foi
usada principalmente para identificar tendências e selecionar a configuração
mais promissora, enquanto a Etapa 5 concentrou o treinamento final e a avaliação
oficial no conjunto de teste.
