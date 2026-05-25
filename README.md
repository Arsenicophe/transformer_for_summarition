# Transformer — Résumé de texte
 
Implémentation from scratch d'un Transformer pour la tâche de résumé automatique de texte.
 
## Architecture
 
Le modèle suit l'architecture encoder-decoder du papier original *Attention Is All You Need* (Vaswani et al., 2017).
 
**Encoder** — lit le texte source et produit une représentation contextuelle de chaque token.
 
<img src="images/encoder_layer.png" alt="Encoder layer" width="330"/>
**Decoder** — génère le résumé token par token, en s'appuyant sur les représentations produites par l'encoder via un mécanisme de cross-attention.
 
<img src="images/decoder_layer.png" alt="Decoder layer" width="330"/>
