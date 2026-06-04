# Checklist final de entrega

Marque cada item somente quando houver evidencia no repositorio ou no relatorio.

- [ ] Repositorio publico ou compartilhado, com README completo.
- [ ] Nomes de todos os integrantes no README e no relatorio.
- [ ] PathMNIST+ oficial 224x224 usado nos pipelines de modelos das etapas 2 a 5; 28x28 restrito a Etapa 1 e a comparacao identica NumPy/PyTorch.
- [ ] Splits oficiais `train`, `val` e `test` preservados.
- [ ] Etapa 1 NumPy com backpropagation, SGD Momentum e gradient check `< 1e-5`.
- [ ] Equivalencia NumPy/PyTorch demonstrada com diferenca final `<= 2 p.p.`.
- [ ] CNN propria com no minimo tres blocos, pooling e dropout.
- [ ] Tres CNNs pre-treinadas comparadas em feature extraction e fine-tuning.
- [ ] ViT pre-treinado comparado com as CNNs.
- [ ] Grid com pelo menos dois otimizadores e tres learning rates.
- [ ] Feature Maps e Grad-CAM de cinco acertos, cinco erros e um caso de atencao errada.
- [ ] Discussao histologica escrita pela equipe.
- [ ] Tecnica XAI adicional executada, se o bonus for reivindicado.
- [ ] Modelo final escolhido estritamente pelo conjunto de validacao.
- [ ] Teste usado uma unica vez no modelo final.
- [ ] Accuracy, F1 macro, F1 por classe e matriz de confusao final salvos.
- [ ] Seeds `random`, `numpy` e `torch` documentadas.
- [ ] Hardware de cada execucao documentado.
- [ ] Dependencias versionadas.
- [ ] Checkpoint do melhor modelo ou link de download.
- [ ] Logs suficientes para reconstruir curvas.
- [ ] Testes `pytest` executados.
- [ ] Relatorio em formato SBC/IEEE, maximo 12 paginas sem referencias.
- [ ] Pelo menos oito referencias, incluindo tres artigos e MedMNIST v2.
- [ ] Uso de IA generativa declarado.
- [ ] Todos os integrantes preparados para explicar qualquer parte do codigo.
