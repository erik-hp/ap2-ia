# Resultados da Etapa 4

## Status atual

Foram recebidos e organizados os seguintes artefatos:

- `outputs/xai/feature_maps_first_conv.png`
- `outputs/xai/gradcam_attention_discussion.png`

Esses arquivos sao visualmente validos como Feature Maps e exemplo de discussao de atencao. Entretanto, a execucao recebida do Colab rodou sem encontrar o checkpoint final, entao ainda nao comprova a explicabilidade do modelo final.

## Pendencias para cumprir a rubrica

- Rodar o notebook 04 com `checkpoints/final/best_final_model.pt` disponivel no Colab/Drive.
- Gerar `outputs/xai/gradcam_correct_high.png` com 5 acertos mais confiantes.
- Gerar `outputs/xai/gradcam_wrong_high.png` com 5 erros mais confiantes.
- Gerar `outputs/xai/xai_selection_summary.json`.
- Escrever a discussao histologica no relatorio, comparando regioes relevantes, fundo e artefatos.

## Ajuste aplicado

O notebook 04 agora exige o checkpoint final e seleciona os exemplos por ranking de confianca, evitando que limiares fixos retornem zero acertos ou erros.
