# Resultados das etapas 1 e 2

Este resumo foi gerado a partir dos outputs salvos nos notebooks `01_numpy_mlp.ipynb` e `02_pytorch_validation.ipynb`.

## Etapa 1 - MLP NumPy

Arquivos de evidencia:

- `experiments/stage01_numpy/numpy_mlp_history.csv`
- `experiments/stage01_numpy/gradient_check.json`
- `experiments/stage01_numpy/numpy_mlp_summary.json`

Resultado final registrado:

| Item | Valor |
|---|---:|
| Epocas | 20 |
| Loss final de treino | 1.591789 |
| Accuracy final de validacao | 0.400140 |
| Gradient check - diferenca relativa maxima | 5.842521e-09 |
| Limite exigido | 1e-05 |

Conclusao: a MLP foi implementada com NumPy puro, incluindo forward, backpropagation, CrossEntropy estavel, SGD com Momentum e validacao numerica. O gradient check ficou abaixo de `1e-5`, atendendo ao criterio obrigatorio.

## Etapa 2 - Validacao PyTorch

Arquivos de evidencia:

- `experiments/stage02_pytorch_validation/torch_mlp_history.csv`
- `experiments/stage02_pytorch_validation/numpy_vs_pytorch_comparison.json`
- `experiments/stage02_pytorch_validation/pathmnist_224_pipeline_check.json`

Resultado final registrado:

| Item | Valor |
|---|---:|
| Accuracy final NumPy | 0.400140 |
| Accuracy final PyTorch | 0.389944 |
| Diferenca absoluta | 1.019593 p.p. |
| Limite exigido | 2.000000 p.p. |

Conclusao: a MLP PyTorch convergiu de forma compativel com a implementacao NumPy no mesmo problema 28x28, com diferenca final menor que 2 pontos percentuais.

## Pipeline 224x224

O pipeline PyTorch tambem foi verificado para entregar tensores no formato exigido para CNNs e backbones pre-treinados:

```text
[16, 3, 224, 224]
```

Para evitar estouro de RAM no Colab, o loader usa PathMNIST 28x28 como fonte leve e aplica `torchvision.transforms.Resize((224, 224))` em tempo de execucao. Assim, os modelos recebem entrada 224x224 sem carregar o arquivo oficial 224x224 completo na memoria.
