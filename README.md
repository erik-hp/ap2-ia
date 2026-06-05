# Projeto Final - IA UniCatolica

Classificacao de tecidos histopatologicos no PathMNIST (MedMNIST v2), com 9 classes e imagens RGB 224x224 nas etapas com PyTorch.

## Equipe e disciplina

- Disciplina: Inteligencia Artificial - UniCatolica
- Equipe: preencher nomes dos integrantes antes da entrega
- Ferramenta de apoio: Codex, usado para estruturacao, revisao de codigo e documentacao

## Escopo

O projeto segue cinco etapas:

1. MLP do zero com NumPy.
2. Validacao equivalente em PyTorch e checagem do pipeline 224x224.
3. CNN propria, CNNs pre-treinadas, ViT e grid de hiperparametros.
4. Explicabilidade com Feature Maps e Grad-CAM.
5. Treino final, avaliacao unica no teste e exportacao de metricas.

## Reprodutibilidade

- Seed global: `42`.
- Dataset: PathMNIST oficial, sem remixar os splits `train`, `val` e `test`.
- Etapa 1: usa PathMNIST 28x28 em grayscale, como permitido para a MLP NumPy.
- Pipelines de modelos das etapas 2 a 5: usam obrigatoriamente o PathMNIST+ oficial 224x224. A comparacao identica NumPy/PyTorch da Etapa 2 preserva os dados 28x28 da Etapa 1. Para caber no Colab, o arquivo oficial 224x224 e extraido em arrays `.npy` e acessado por memoria mapeada, sem carregar o dataset inteiro na RAM.
- O test set deve ser executado uma unica vez, apenas no modelo final escolhido pelo validation set.

Instalacao local:

```bash
pip install -r requirements.txt
```

Instalacao no Colab:

```bash
pip install -r requirements-colab.txt
```

## Ordem de execucao

Execute os notebooks em ordem:

1. `notebooks/01_numpy_mlp.ipynb`
2. `notebooks/02_pytorch_validation.ipynb`
3. `notebooks/03_cnns_and_vit.ipynb`
4. `notebooks/05_final_model.ipynb`
5. `notebooks/04_xai.ipynb`

O notebook 04 depende do checkpoint final salvo pelo notebook 05.

## Artefatos gerados

Os notebooks salvam resultados suficientes para auditar curvas, escolha de modelo e avaliacao final.

| Etapa | Arquivos principais |
|---|---|
| 1 | `experiments/stage01_numpy/numpy_mlp_history.csv`, `gradient_check.json`, `outputs/figures/stage01_numpy_curves.png` |
| 2 | `experiments/stage02_pytorch_validation/torch_mlp_history.csv`, `numpy_vs_pytorch_comparison.json`, `pathmnist_224_pipeline_check.json` |
| 3 | `experiments/results.csv`, `stage03_experiment_manifest.json`, `stage03_best_by_run.csv`, `outputs/figures/stage03_validation_curves.png` |
| 4 | `outputs/xai/gradcam_correct_high.png`, `gradcam_wrong_high.png`, `gradcam_attention_discussion.png`, `feature_maps_first_conv.png`, `xai_selection_summary.json` |
| 5 | `checkpoints/final/best_final_model.pt`, `experiments/final/final_training_history.csv`, `test_metrics.json`, `test_classification_report.csv`, `test_confusion_matrix.csv` |

Os resultados ja executados das etapas 1 e 2 estao resumidos em `docs/RESULTADOS_ETAPAS_01_02.md`.
Os resultados da etapa 3 estao resumidos em `docs/RESULTADOS_ETAPA_03.md`.
O status da etapa 4 esta resumido em `docs/RESULTADOS_ETAPA_04.md`.
Os resultados da etapa 5 estao resumidos em `docs/RESULTADOS_ETAPA_05.md`.

Checkpoints `.pt` nao devem ser versionados se ficarem grandes. Para a entrega final, envie o arquivo junto com o trabalho ou informe um link de download no relatorio.

## Hardware utilizado

Preencher antes da entrega final, com base no ambiente usado nas execucoes. Os logs da Etapa 3 e da Etapa 5 registram tempo e VRAM por experimento, mas o arquivo global `experiments/hardware.json` nao esta versionado.

- CPU:
- GPU:
- RAM:
- VRAM:

## Comandos uteis

Treino de um experimento:

```bash
python src/train.py --model resnet50 --mode feature_extraction --optimizer adamw --lr 1e-3 --epochs 10 --batch-size 32 --image-size 224 --source-size 224 --results experiments/results.csv --checkpoint checkpoints/stage03/resnet50_feature_extraction.pt
```

Treino final com regularizacao e scheduler:

```bash
python src/train.py --model resnet50 --mode fine_tuning --optimizer adamw --lr 1e-4 --epochs 30 --batch-size 16 --image-size 224 --source-size 224 --augment-policy randaugment --label-smoothing 0.1 --cosine --early-stopping --patience 5 --checkpoint checkpoints/final/best_final_model.pt
```

Rastreamento opcional com WandB:

```bash
python src/train.py --model resnet50 --mode fine_tuning --use-wandb --run-name resnet50-ft
```

## Testes

Execute a suite sem baixar dataset ou pesos:

```bash
pytest
```

## Aderencia aos criterios

O mapeamento detalhado da rubrica esta em `docs/CRITERIOS.md`.
O roteiro do artigo esta em `docs/RELATORIO_MODELO.md`.
A auditoria de conformidade e pendencias esta em `docs/AUDITORIA_REQUISITOS.md`.
O checklist de entrega esta em `docs/CHECKLIST_ENTREGA.md`.

Resumo:

- Corretude NumPy: MLP, CrossEntropy estavel, SGD Momentum e gradient checking em `src/numpy_nn/`.
- Pipelines e modelagem: `src/data/`, `src/models/`, `src/train.py` e notebook 03.
- XAI: `src/xai/` e notebook 04.
- Rigor e artigo: logs CSV/JSON, checkpoints, hardware, matriz de confusao e protocolo de teste unico.

## Declaracao de uso de IA generativa

Ferramenta utilizada: Codex.

Finalidade: apoio na estruturacao do repositorio, implementacao-base dos modelos, utilitarios de treino, explicabilidade, auditoria de requisitos e documentacao. A IA nao deve ser usada para redigir o artigo final, exceto para revisao gramatical permitida. As decisoes experimentais, resultados, discussoes clinicas e conclusoes devem ser produzidas e validadas pela equipe apos execucao dos notebooks.

Interacoes relevantes:

| Etapa | Uso da IA |
|---|---|
| Estruturacao | Organizacao inicial de `src/`, notebooks, artefatos e documentacao |
| Implementacao | Revisao de robustez, loader oficial 224x224 com memoria mapeada, testes e WandB opcional |
| Auditoria | Comparacao do repositorio com os criterios oficiais e identificacao de lacunas |
| Escrita | Revisao estrutural do README e criacao de modelos/checklists; a analise cientifica final permanece responsabilidade da equipe |

## Link do relatorio PDF

Preencher link final.
