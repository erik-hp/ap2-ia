# Mapa do Código e Evidências

Este arquivo serve como guia rápido para localizar cada parte do projeto durante
a correção. O código principal fica em `src/`, os experimentos e métricas ficam
em `experiments/`, as figuras ficam em `outputs/` e os notebooks documentam a
execução etapa por etapa.

## Etapa 1 - MLP em NumPy

- Implementação da rede: `src/numpy_nn/model.py`
- Softmax e Cross-Entropy: `src/numpy_nn/losses.py`
- SGD com Momentum: `src/numpy_nn/optimizers.py`
- Gradient check: `src/numpy_nn/gradient_check.py`
- Notebook: `notebooks/01_numpy_mlp.ipynb`
- Evidências: `experiments/stage01_numpy/`

## Etapa 2 - Validação em PyTorch

- Pipeline de dados: `src/data/dataset.py`
- Notebook: `notebooks/02_pytorch_validation.ipynb`
- Comparação NumPy vs PyTorch:
  `experiments/stage02_pytorch_validation/numpy_vs_pytorch_comparison.json`

## Etapa 3 - CNNs, Transfer Learning e ViT

- Script de treino e logging: `src/train.py`
- CNN autoral: `src/models/custom_cnn.py`
- Modelos pré-treinados: `src/models/transfer.py`
- Resultados comparativos: `experiments/stage03_best_by_run.csv`
- Curvas de validação: `outputs/figures/stage03_validation_curves.png`
- Notebook: `notebooks/03_cnns_and_vit.ipynb`

## Etapa 4 - Explainable AI

- Feature Maps: `src/xai/feature_maps.py`
- Grad-CAM: `src/xai/gradcam.py`
- Notebook: `notebooks/04_xai.ipynb`
- Figuras:
  - `outputs/xai/feature_maps_first_conv.png`
  - `outputs/xai/gradcam_correct_high.png`
  - `outputs/xai/gradcam_wrong_high.png`
  - `outputs/xai/gradcam_attention_discussion.png`

## Etapa 5 - Treinamento Final e Teste

- Configuração final: `experiments/final/final_training_config.json`
- Histórico final: `experiments/final/final_training_history.csv`
- Métricas de teste: `experiments/final/test_metrics.json`
- Classification report: `experiments/final/test_classification_report.csv`
- Matriz de confusão: `experiments/final/test_confusion_matrix.csv`
- Figuras:
  - `outputs/figures/stage05_final_training_curves.png`
  - `outputs/figures/stage05_test_confusion_matrix.png`
- Notebook: `notebooks/05_final_model.ipynb`

## Arquivos de apoio

- Reprodutibilidade, seed, hardware e CSV: `src/utils.py`
- Exportação de métricas e matriz de confusão: `src/evaluate.py`
- Auditoria de requisitos: `docs/AUDITORIA_REQUISITOS.md`
- Texto-base do relatório: `docs/RELATORIO_MODELO.md`
