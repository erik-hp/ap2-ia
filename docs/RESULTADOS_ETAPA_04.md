# Resultados da Etapa 4

## Status

A Etapa 4 agora possui os artefatos principais de explicabilidade gerados a partir do checkpoint final local (`checkpoints/final/best_final_model.pt`):

- `outputs/xai/feature_maps_first_conv.png`
- `outputs/xai/gradcam_correct_high.png`
- `outputs/xai/gradcam_wrong_high.png`
- `outputs/xai/gradcam_attention_discussion.png`
- `outputs/xai/xai_selection_summary.json`

## Evidencias geradas

O arquivo `outputs/xai/xai_selection_summary.json` registra:

- 5 predicoes corretas selecionadas para Grad-CAM.
- 5 predicoes erradas selecionadas para Grad-CAM.
- 1 exemplo separado para discutir atencao em regiao possivelmente inadequada.
- Confiancas dos acertos: 0.9962, 0.9949, 0.9931, 0.9901 e 0.9883.
- Confiancas dos erros: 0.8556, 0.7284, 0.6407, 0.5591 e 0.3638.

Os mapas de ativacao da primeira camada convolucional foram salvos em grade com 16 filtros, permitindo discutir se os filtros iniciais capturam bordas, textura, padroes nucleares e variacoes de intensidade.

## Avaliacao contra a rubrica

- Feature Maps: atendido pelo arquivo `feature_maps_first_conv.png`.
- Grad-CAM em 5 acertos de alta confianca: atendido pelo arquivo `gradcam_correct_high.png`.
- Grad-CAM em 5 erros: atendido pelo arquivo `gradcam_wrong_high.png`. Observacao: os erros foram selecionados por ranking de confianca; os primeiros sao bem confiantes, mas o quinto erro tem confianca moderada/baixa.
- Caso de atencao inadequada para discussao: atendido pelo arquivo `gradcam_attention_discussion.png`.
- Discussao clinica/histologica: precisa ser escrita no relatorio final, usando os artefatos acima como evidencia.

## Pontos de atencao

O arquivo `gradcam_attention_discussion.png` contem o exemplo solicitado, mas o texto do titulo ficou visualmente apertado. Para o relatorio, recomenda-se usar a imagem como apoio e escrever a interpretacao no texto, em vez de depender apenas do titulo da figura.

Tecnica extra de bonus, como SHAP, Integrated Gradients ou Attention Rollout, nao foi incluida. Isso nao impede o cumprimento da Etapa 4 basica, mas nao deve ser declarado como realizado.
