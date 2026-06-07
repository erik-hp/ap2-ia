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
- `experiments/final/test_classification_report.csv`
- `experiments/final/test_confusion_matrix.csv`
- `outputs/figures/stage05_final_training_curves.png`
- `outputs/figures/stage05_test_confusion_matrix.png`
- `checkpoints/final/best_final_model.pt` localmente.

## Checkpoint

O checkpoint final foi recuperado para `checkpoints/final/best_final_model.pt`.
Arquivos `.pt` continuam ignorados pelo Git por tamanho. O checkpoint final está
disponível em:

https://drive.google.com/drive/folders/1hI9ir_JGPJxQYuY1zgNgXdfsCyJy-HD8?usp=sharing
