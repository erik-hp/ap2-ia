# Resultados da Etapa 3

Execucao rapida realizada no Kaggle com PathMNIST+ oficial 224x224, `batch_size=16` e `epochs=1`, devido aos limites de GPU no Colab/Kaggle. O objetivo desta etapa foi cobrir a matriz comparativa exigida pela rubrica, registrando acuracia de validacao, tempo e VRAM, sem declarar convergencia total.

## Cobertura dos requisitos

- CNN propria: `custom_cnn`.
- Tres CNNs pre-treinadas: `resnet50`, `efficientnet_b0`, `mobilenet_v3_large`.
- Vision Transformer: `vit_b_16`.
- Feature extraction e fine-tuning presentes.
- Grid no `resnet50` com 2 otimizadores (`sgd`, `adamw`) x 3 learning rates (`1e-2`, `1e-3`, `1e-4`).
- Logs salvos em `experiments/results.csv`.
- Melhor modelo por validacao salvo em `experiments/stage03_best_by_run.csv`.
- Curvas salvas em `outputs/figures/stage03_validation_curves.png`.

## Melhor configuracao por validacao

| tag | modelo | modo | otimizador | lr | acc_val | tempo_s | vram_mb |
|---|---|---|---|---:|---:|---:|---:|
| `resnet50_fine_tuning_sgd_1e2` | `resnet50` | `fine_tuning` | `sgd` | 0.01 | 0.992903 | 1152.66 | 1653.36 |

## Observacao metodologica

Os valores foram recuperados do output salvo do notebook Kaggle porque o push direto de artefatos pelo Kaggle falhou. Os CSVs e metadados foram reconstruidos sem alterar os numeros exibidos no notebook executado.
