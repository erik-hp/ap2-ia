# Auditoria dos requisitos oficiais

Auditoria baseada no PDF integral da atividade e no texto complementar fornecido. O status distingue implementacao de evidencia experimental: codigo preparado nao substitui execucao e analise da equipe.

## Implementado no repositorio

| Requisito | Evidencia |
|---|---|
| MLP somente NumPy, backpropagation, CrossEntropy estavel e SGD Momentum | `src/numpy_nn/`, notebook 01 |
| Gradient check `< 1e-5` | `experiments/stage01_numpy/gradient_check.json` |
| Equivalencia NumPy/PyTorch `<= 2 p.p.` nos mesmos dados da Etapa 1 | `experiments/stage02_pytorch_validation/numpy_vs_pytorch_comparison.json` |
| PathMNIST+ oficial 224x224 nas etapas PyTorch | `MemoryMappedPathMNIST` em `src/data/dataset.py` |
| Splits oficiais preservados | `get_loaders()` cria loaders separados sem remix |
| CNN propria com tres blocos, pooling e dropout | `src/models/custom_cnn.py` |
| Tres CNNs pre-treinadas e ViT | `src/models/transfer.py`, notebook 03 |
| Feature extraction e fine-tuning | `create_model()` e notebook 03 |
| Grid dois otimizadores x tres LRs | notebook 03 |
| Scheduler, regularizacoes e augmentacao avancada | `src/train.py`, `src/data/dataset.py`, notebook 05 |
| Feature Maps e Grad-CAM | `src/xai/`, notebook 04 |
| Teste final bloqueado contra reexecucao acidental | notebook 05 |
| Logs CSV/JSON, hardware e checkpoints | `src/utils.py`, `src/train.py`, notebooks |
| Checkpoint configuravel por metrica/modo | argumentos `--checkpoint-metric` e `--checkpoint-mode` |
| WandB opcional | argumento `--use-wandb` |
| Testes unitarios e CI | `tests/`, `pytest.ini`, `.github/workflows/tests.yml` |
| Declaracao de IA | `README.md` |

## Evidencias ainda pendentes

- A celula final do notebook 02 deve ser reexecutada com o loader oficial 224x224. O codigo do notebook ja usa `source_size=224`, mas o JSON versionado ainda e historico e esta marcado como `criterion_met: false`.
- Documentar o hardware real usado nas execucoes finais. `experiments/results.csv` e `experiments/final/final_training_history.csv` registram tempo e VRAM, mas `experiments/hardware.json` nao esta presente.
- Disponibilizar o checkpoint final ou link de download no relatorio/entrega, pois arquivos `.pt` nao sao versionados no Git.

## Itens obrigatoriamente manuais

- Preencher nomes dos integrantes.
- Escrever discussao histologica e conclusoes cientificas pela equipe.
- Produzir artigo SBC/IEEE com no maximo 12 paginas sem referencias.
- Incluir pelo menos oito referencias, sendo tres artigos originais e MedMNIST v2 obrigatorio.
- Disponibilizar checkpoint final ou link de download.
- Preparar todos os integrantes para explicar qualquer parte do codigo na apresentacao oral.

## Observacao metodologica critica

A comparacao de validacao NumPy/PyTorch da Etapa 2 preserva os mesmos dados
28x28 da Etapa 1, como o proprio criterio exige. Separadamente, todos os
pipelines destinados aos modelos das etapas 2 a 5 usam a fonte oficial
224x224. O loader mapeado em disco foi criado especificamente para cumprir
essa regra sem esgotar a RAM do Colab.
