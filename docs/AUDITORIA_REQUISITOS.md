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
| Feature Maps, Grad-CAM e Integrated Gradients | `src/xai/`, notebook 04 |
| Teste final bloqueado contra reexecucao acidental | notebook 05 |
| Logs CSV/JSON, hardware e checkpoints | `src/utils.py`, `src/train.py`, notebooks |
| Checkpoint configuravel por metrica/modo | argumentos `--checkpoint-metric` e `--checkpoint-mode` |
| WandB opcional | argumento `--use-wandb` |
| Testes unitarios e CI | `tests/`, `pytest.ini`, `.github/workflows/tests.yml` |
| Declaracao de IA | `README.md` |

## Evidencias que precisam ser reexecutadas

- A celula final do notebook 02 deve ser reexecutada com o novo loader oficial 224x224. O JSON historico esta marcado como nao conforme.
- O notebook 03 deve ser executado para preencher `experiments/results.csv`, checkpoints, tempo e VRAM.
- O notebook 05 deve ser executado apos a escolha estrita pelo validation set e avaliar o teste uma unica vez.
- O notebook 04 deve ser executado depois do checkpoint final para gerar as imagens XAI.

## Itens obrigatoriamente manuais

- Preencher nomes dos integrantes.
- Documentar hardware real de cada execucao.
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
