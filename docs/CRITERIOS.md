# Checklist de aderencia aos criterios

Este arquivo liga cada exigencia da rubrica aos arquivos do projeto. Os resultados numericos so ficam completos depois da execucao dos notebooks em ordem.

## FAQ e restricoes

- Etapa 1 usa apenas NumPy para a MLP: `src/numpy_nn/` e `notebooks/01_numpy_mlp.ipynb`.
- Etapas 2 a 5 usam PyTorch/torchvision: `src/train.py`, `src/data/dataset.py`, `src/models/`.
- PathMNIST 28x28 e usado apenas na comparacao algoritmica da Etapa 1 e na validacao equivalente da MLP PyTorch. O pipeline 224x224 e verificado no notebook 02 e usado nas etapas 3 a 5.
- Treino em Colab/Kaggle e suportado pelos notebooks com instalacao de dependencias ausentes e deteccao da raiz do projeto.

## Reprodutibilidade

- Seeds: `utils.set_seed(42)` sincroniza `random`, `numpy` e `torch`.
- Dependencias: `requirements.txt` e `requirements-colab.txt`.
- Hardware: gerado em `experiments/hardware.json` pelos notebooks 03 e 05.
- Logs de treino: `experiments/results.csv`, `experiments/stage01_numpy/*.csv`, `experiments/final/*.csv`.
- Checkpoints: `checkpoints/stage03/*.pt` e `checkpoints/final/best_final_model.pt`.
- Figuras: `outputs/figures/*.png` e `outputs/xai/*.png`.

## Etapas

1. MLP NumPy: `notebooks/01_numpy_mlp.ipynb`, `src/numpy_nn/`, `experiments/stage01_numpy/gradient_check.json`.
2. Validacao PyTorch: `notebooks/02_pytorch_validation.ipynb`, `experiments/stage02_pytorch_validation/numpy_vs_pytorch_comparison.json`, `pathmnist_224_pipeline_check.json`.
3. CNNs, ViT e grid: `notebooks/03_cnns_and_vit.ipynb`, `src/models/`, `experiments/stage03_experiment_manifest.json`, `experiments/stage03_best_by_run.csv`.
4. XAI: `notebooks/04_xai.ipynb`, `src/xai/`, `outputs/xai/`.
5. Modelo final: `notebooks/05_final_model.ipynb`, `experiments/final/test_metrics.json`, `test_classification_report.csv`, `test_confusion_matrix.csv`.

## Criterio do teste unico

O notebook 05 cria `experiments/final/test_evaluation_protocol.json` na avaliacao final. Se `test_metrics.json` ja existir, a celula bloqueia nova avaliacao do teste por padrao. Para reavaliar, e necessario mudar explicitamente `ALLOW_REEVALUATE_TEST=True` e justificar no relatorio.
