# Modelo de estrutura do relatorio

O PDF final deve seguir formato de artigo cientifico, modelo SBC ou IEEE, com no maximo 12 paginas sem contar referencias.

## 1. Introducao

- Contextualizacao do problema histopatologico.
- Importancia clinica e objetivo de classificacao.
- Justificativa da abordagem.
- Resumo das contribuicoes.

## 2. Fundamentacao teorica

- Backpropagation.
- SGD com Momentum.
- CNNs, transfer learning e fine-tuning.
- Vision Transformers.
- Explicabilidade com Feature Maps, Grad-CAM e tecnica adicional.

## 3. Metodologia

- Estrutura do codigo em NumPy e PyTorch.
- PathMNIST+ oficial 224x224, nove classes e splits oficiais.
- Arquiteturas, modos de treino e hiperparametros.
- Metodologia de busca no conjunto de validacao.
- Hardware, seeds, dependencias e criterios de reproducibilidade.

## 4. Resultados e discussao

- Curvas de loss/accuracy de treino e validacao.
- Comparacao NumPy vs PyTorch.
- Comparacao CNN propria, tres CNNs pre-treinadas e ViT.
- Tabela de accuracy, tempo e VRAM.
- Feature Maps, Grad-CAM e discussao histologica.
- Accuracy, F1 macro, F1 por classe e matriz de confusao do teste final.

## 5. Conclusao

- Sintese das descobertas.
- Reflexao sobre custo-beneficio entre arquiteturas.
- Limitacoes e trabalhos futuros.

## 6. Referencias

- Minimo de oito referencias.
- Pelo menos tres artigos cientificos originais.
- Incluir obrigatoriamente MedMNIST v2.

## Declaracoes obrigatorias

- Uso de IA generativa e finalidade.
- Hardware de cada execucao.
- Seed utilizada.
- Declaracao de que o conjunto de teste foi usado uma unica vez no modelo final.
