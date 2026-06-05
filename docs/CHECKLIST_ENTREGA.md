# Checklist final de entrega

Marque cada item somente quando houver evidencia no repositorio ou no relatorio.

- [x] Repositorio publico ou compartilhado, com README estrutural.
- [ ] Nomes de todos os integrantes no README e no relatorio.
- [ ] PathMNIST+ oficial 224x224 usado nos pipelines das etapas 2 a 5; pendente atualizar o JSON da checagem 224 da Etapa 2, que ainda esta historico.
- [x] Splits oficiais `train`, `val` e `test` preservados.
- [x] Etapa 1 NumPy com backpropagation, SGD Momentum e gradient check `< 1e-5`.
- [x] Equivalencia NumPy/PyTorch demonstrada com diferenca final `<= 2 p.p.`.
- [x] CNN propria com no minimo tres blocos, pooling e dropout.
- [x] Tres CNNs pre-treinadas comparadas em feature extraction e fine-tuning.
- [x] ViT pre-treinado comparado com as CNNs.
- [x] Grid com pelo menos dois otimizadores e tres learning rates.
- [x] Feature Maps e Grad-CAM de cinco acertos, cinco erros e um caso de atencao errada.
- [ ] Discussao histologica escrita pela equipe.
- [ ] Tecnica XAI adicional executada, se o bonus for reivindicado.
- [x] Modelo final escolhido estritamente pelo conjunto de validacao.
- [x] Teste usado uma unica vez no modelo final.
- [x] Accuracy, F1 macro, F1 por classe e matriz de confusao final salvos.
- [x] Seeds `random`, `numpy` e `torch` documentadas.
- [ ] Hardware de cada execucao documentado.
- [x] Dependencias versionadas.
- [ ] Checkpoint do melhor modelo ou link de download.
- [x] Logs suficientes para reconstruir curvas.
- [ ] Testes `pytest` executados no ambiente de entrega; nao rodaram localmente porque `pytest` nao esta instalado neste Python.
- [ ] Relatorio em formato SBC/IEEE, maximo 12 paginas sem referencias.
- [ ] Pelo menos oito referencias, incluindo tres artigos e MedMNIST v2.
- [x] Uso de IA generativa declarado.
- [ ] Todos os integrantes preparados para explicar qualquer parte do codigo.
