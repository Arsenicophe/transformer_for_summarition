from decoder import Decoder
from encoder import Encoder
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(
        self,
        embedding_dim,
        vocab_size,
        n_heads,
        n_layer,
        max_seq_encoding,
        dropout,
    ):
        super().__init__()

        self.encoder = Encoder(
            embedding_dim, vocab_size, n_layer,
            n_heads, max_seq_encoding, dropout,
        )

        self.decoder = Decoder(
            embedding_dim, vocab_size, n_heads,
            n_layer, max_seq_encoding, dropout,
        )

        # ── Projection finale vers le vocabulaire ─────────────────────────────
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # ── Weight tying ──────────────────────────────────────────────────────
        # On partage 3 matrices qui représentent tous le même espace tokenizer :
        #
        #   encoder.embedding   : token → vecteur  (côté source)
        #   decoder.embedding   : token → vecteur  (côté cible)
        #   lm_head             : vecteur → logits (prédiction)
        #
        # Même vocabulaire → même géométrie → partager les poids est cohérent.
        # Avantage : -25M paramètres, meilleure généralisation.
        #
        #   Sans tying : 3 × (50265 × 256) = 38.6M params juste en embeddings
        #   Avec tying : 1 × (50265 × 256) = 12.9M params  (-67%)

        self.decoder.embedding.weight = self.encoder.embedding.weight   # enc ↔ dec
        self.lm_head.weight           = self.decoder.embedding.weight   # dec ↔ lm_head

        # ── Final LayerNorm ───────────────────────────────────────────────────
        # Nécessaire avec Pre-LN : le dernier bloc ne passe plus par un LN
        # avant la projection → on en ajoute un explicitement.
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        enc_idx,
        input_idx,
        enc_mask_padding,
        causal_mask,
    ):
        context = self.encoder(enc_idx, enc_mask_padding)
        decoded = self.decoder(input_idx, context, enc_mask_padding, causal_mask)
        decoded = self.final_norm(decoded)          # ← final LN avant projection
        return self.lm_head(decoded)