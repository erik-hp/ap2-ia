# Projeto Final - Inteligência Artificial

Classificação de tecidos histopatológicos usando redes neurais profundas no
dataset PathMNIST, com comparação entre MLP, CNN própria, modelos
pré-treinados, Vision Transformer e técnicas de Explainable AI.

## Identificação

- Disciplina: Inteligência Artificial
- Instituição: Centro Universitário Católica de Quixadá - UniCatólica
- Equipe:
  - Erik Holanda Pires
  - Carlos Daniel Reis da Silva
  - Raul Ferreira Holanda
- Dataset: PathMNIST, da coleção MedMNIST v2
- Tarefa: classificação multiclasse de nove tipos de tecidos histopatológicos

## Objetivo do projeto

O objetivo é implementar, comparar e analisar diferentes arquiteturas de redes
neurais para classificar imagens histopatológicas do PathMNIST. O projeto também
busca demonstrar interpretabilidade das predições com Feature Maps e Grad-CAM,
permitindo observar quais regiões das imagens influenciaram as decisões do
modelo final.

## Resumo das etapas

| Etapa | Descrição | Principal evidência |
|---|---|---|
| 1 | MLP implementada do zero com NumPy | `src/numpy_nn/`, `notebooks/01_numpy_mlp.ipynb` |
| 2 | Reprodução da MLP em PyTorch para validar a implementação NumPy | `notebooks/02_pytorch_validation.ipynb` |
| 3 | Comparação entre CNN própria, ResNet50, EfficientNet-B0, MobileNetV3 Large e ViT-B/16 | `notebooks/03_cnns_and_vit.ipynb` |
| 4 | Interpretabilidade com Feature Maps e Grad-CAM | `notebooks/04_xai.ipynb`, `outputs/xai/` |
| 5 | Treinamento final e avaliação única no conjunto de teste | `notebooks/05_final_model.ipynb`, `experiments/final/` |

## Resultados principais

- Melhor modelo na validação: ResNet50 em modo fine-tuning.
- Melhor acurácia de validação na etapa comparativa: `99,29%`.
- Modelo final: ResNet50 fine-tuning com SGD, learning rate `0.01`,
  RandAugment, label smoothing `0.1`, batch size `16` e seed `42`.
- Resultado no conjunto oficial de teste:
  - Accuracy: `96,71%`
  - F1-Score Macro: `95,41%`
- Técnicas de interpretabilidade aplicadas:
  - Feature Maps da primeira camada convolucional.
  - Grad-CAM em predições corretas, incorretas e em caso de atenção inadequada.

## Organização do repositório

```text
AP2_IA/
+-- src/
|   +-- data/              # Dataset, transformações e DataLoaders oficiais
|   +-- models/            # CNN própria e modelos de transfer learning
|   +-- numpy_nn/          # MLP NumPy, loss, otimizador e gradient check
|   +-- xai/               # Feature Maps e Grad-CAM
|   +-- train.py           # Treinamento, validação, logging e checkpoint
|   +-- evaluate.py        # Métricas, classification report e matriz de confusão
|   +-- utils.py           # Seed, hardware, CSV/JSON, timer e early stopping
+-- notebooks/             # Execução organizada por etapa
+-- experiments/           # Resultados numéricos em CSV/JSON
+-- outputs/
|   +-- figures/           # Curvas e matriz de confusão
|   +-- xai/               # Figuras de interpretabilidade
+-- tests/                 # Testes automatizados do projeto
+-- docs/                  # Guias, auditoria, critérios e resumos de resultados
+-- requirements*.txt      # Dependências local e Colab
```

Para uma navegação ainda mais direta, consulte `docs/MAPA_DO_CODIGO.md`. Esse
arquivo mostra onde cada requisito aparece no código, nos notebooks e nos
artefatos gerados.

## Reprodutibilidade

- Seed global usada nos experimentos: `42`.
- Splits oficiais preservados: `train`, `val` e `test`.
- O conjunto de teste é usado apenas uma vez, na avaliação final.
- A Etapa 1 usa imagens `28x28` em escala de cinza para a MLP NumPy.
- As etapas PyTorch usam o PathMNIST+ oficial `224x224`.
- O arquivo oficial `pathmnist_224.npz` é extraído em arrays `.npy` e lido com
  memória mapeada, evitando carregar todo o dataset na RAM.
- Os resultados são salvos em CSV/JSON para permitir auditoria posterior.

## Como instalar

Ambiente local:

```bash
pip install -r requirements.txt
```

Google Colab:

```bash
pip install -r requirements-colab.txt
```

## Ordem recomendada de execução

Execute os notebooks nesta ordem:

1. `notebooks/01_numpy_mlp.ipynb`
2. `notebooks/02_pytorch_validation.ipynb`
3. `notebooks/03_cnns_and_vit.ipynb`
4. `notebooks/05_final_model.ipynb`
5. `notebooks/04_xai.ipynb`

O notebook 04 deve ser executado depois do notebook 05, pois depende do
checkpoint final salvo pelo treinamento final.

## Onde conferir cada resultado

| Conteúdo | Arquivo ou pasta |
|---|---|
| Resultado da MLP NumPy | `experiments/stage01_numpy/` |
| Comparação NumPy vs PyTorch | `experiments/stage02_pytorch_validation/` |
| Comparação de arquiteturas | `experiments/stage03_best_by_run.csv` |
| Curvas de validação da etapa 3 | `outputs/figures/stage03_validation_curves.png` |
| Configuração do modelo final | `experiments/final/final_training_config.json` |
| Histórico do treino final | `experiments/final/final_training_history.csv` |
| Métricas finais de teste | `experiments/final/test_metrics.json` |
| Classification report | `experiments/final/test_classification_report.csv` |
| Matriz de confusão | `experiments/final/test_confusion_matrix.csv` |
| Figura da matriz de confusão | `outputs/figures/stage05_test_confusion_matrix.png` |
| Feature Maps | `outputs/xai/feature_maps_first_conv.png` |
| Grad-CAM | `outputs/xai/gradcam_correct_high.png`, `outputs/xai/gradcam_wrong_high.png` |

## Documentos de apoio

- `docs/MAPA_DO_CODIGO.md`: guia rápido de onde está cada parte do projeto.
- `docs/CRITERIOS.md`: mapeamento dos critérios da atividade.
- `docs/AUDITORIA_REQUISITOS.md`: auditoria de conformidade e pendências.
- `docs/RESULTADOS_ETAPAS_01_02.md`: resumo das etapas 1 e 2.
- `docs/RESULTADOS_ETAPA_03.md`: resumo da comparação de arquiteturas.
- `docs/RESULTADOS_ETAPA_04.md`: resumo da etapa de XAI.
- `docs/RESULTADOS_ETAPA_05.md`: resumo do treino final e teste.
- `docs/RELATORIO_MODELO.md`: roteiro de escrita do artigo.

## Comandos úteis

Treinar um experimento da etapa 3:

```bash
python src/train.py --model resnet50 --mode feature_extraction --optimizer adamw --lr 1e-3 --epochs 10 --batch-size 32 --image-size 224 --source-size 224 --results experiments/results.csv --checkpoint checkpoints/stage03/resnet50_feature_extraction.pt
```

Treinar o modelo final:

```bash
python src/train.py --model resnet50 --mode fine_tuning --optimizer sgd --lr 0.01 --epochs 5 --batch-size 16 --image-size 224 --source-size 224 --augment-policy randaugment --label-smoothing 0.1 --checkpoint checkpoints/final/best_final_model.pt
```

Executar os testes:

```bash
pytest
```

## Observações sobre checkpoints

Arquivos `.pt` podem ficar grandes e normalmente não são versionados no Git.
Quando o checkpoint final for exigido, ele deve acompanhar os demais arquivos da
entrega.

## Declaração de uso de IA generativa

Ferramenta utilizada: Codex.

Finalidade: apoio na estruturação do repositório, revisão de código,
documentação, organização de evidências e auditoria dos requisitos. As decisões
experimentais, a interpretação dos resultados e a versão final do texto
científico devem ser revisadas e validadas pela equipe.

| Etapa | Uso da IA |
|---|---|
| Estruturação | Organização inicial de `src/`, notebooks, artefatos e documentação |
| Implementação | Revisão de robustez, loader oficial `224x224`, testes e utilitários |
| Auditoria | Comparação do repositório com os critérios da atividade |
| Escrita | Apoio na clareza da documentação e revisão estrutural |
