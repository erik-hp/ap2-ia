# Projeto Final - IA UniCatolica

Classificacao de tecidos histopatologicos no PathMNIST (MedMNIST v2), com 9 classes e imagens RGB 224x224.

## Equipe e disciplina

- Disciplina: Inteligencia Artificial - UniCatolica
- Equipe: preencher nomes dos integrantes

## Hardware utilizado

Preencher apos execucao:

- CPU:
- GPU:
- RAM:
- VRAM:

## Reprodutibilidade

- Seed global: `42`
- Dataset: PathMNIST oficial, sem remixar os splits `train`, `val` e `test`
- O test set deve ser executado uma unica vez, apenas no modelo final escolhido pelo validation set.

Instalacao:

```bash
pip install -r requirements.txt
```

Ordem sugerida dos notebooks:

1. `notebooks/01_numpy_mlp.ipynb`
2. `notebooks/02_pytorch_validation.ipynb`
3. `notebooks/03_cnns_and_vit.ipynb`
4. `notebooks/04_xai.ipynb`
5. `notebooks/05_final_model.ipynb`

Exemplo de experimento:

```bash
python src/train.py --model resnet50 --mode feature_extraction --optimizer adamw --lr 1e-3 --epochs 10
```

Exemplo do treino final:

```bash
python src/train.py --model resnet50 --mode fine_tuning --optimizer adamw --lr 1e-4 --epochs 30 --label-smoothing 0.1 --cosine --early-stopping --patience 5
```

## Declaracao de uso de IA generativa

Ferramenta utilizada: Codex.

Finalidade: apoio na estruturacao inicial do repositorio, implementacao-base dos modelos, utilitarios de treino, explicabilidade e documentacao. As decisoes experimentais, resultados, analises e conclusoes devem ser validadas pela equipe.

## Link do relatorio PDF

Preencher link final.

teste - daniel