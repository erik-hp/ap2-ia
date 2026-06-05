# Resultados da Etapa 5

Execucao final realizada com o melhor modelo escolhido pela Etapa 3:

- Modelo: `resnet50`
- Modo: `fine_tuning`
- Otimizador: `sgd`
- Learning rate inicial: `0.01`
- Epocas: `5`
- Batch size: `16`
- Imagem: PathMNIST+ oficial `224x224`
- Augmentacao: `randaugment`
- Label smoothing: `0.1`

## Validacao

O melhor checkpoint foi selecionado por `val_loss` na epoca 3:

| epoca | loss_val | acc_val |
|---:|---:|---:|
| 3 | 0.523702 | 0.992603 |

## Teste final

O notebook avaliou o conjunto de teste uma unica vez e registrou:

| metrica | valor |
|---|---:|
| accuracy | 0.967131 |
| f1_macro | 0.954083 |

## Artefatos presentes

- `experiments/final/final_training_config.json`
- `experiments/final/final_training_history.csv`
- `experiments/final/test_metrics.json`
- `experiments/final/test_evaluation_protocol.json`
- `outputs/figures/stage05_final_training_curves.png`
- `outputs/figures/stage05_test_confusion_matrix.png`

## Pendencias

Os seguintes arquivos foram gerados pelo notebook, mas nao vieram no commit do Kaggle e precisam ser baixados/salvos para a entrega ficar completa:

- `experiments/final/test_classification_report.csv`
- `experiments/final/test_confusion_matrix.csv`
- `checkpoints/final/best_final_model.pt` ou link externo para o checkpoint.

O checkpoint nao precisa necessariamente entrar no GitHub; pode ser disponibilizado por link de download e citado no README/relatorio.
