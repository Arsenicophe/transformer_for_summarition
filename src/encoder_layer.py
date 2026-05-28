from torch import nn


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, enc_input, mask_padding=None, need_weights=False):

        # ── Pre-LayerNorm : normaliser AVANT le sous-module ──────────────────
        # Post-LN (ancien) : x = LayerNorm(x + SubLayer(x))
        # Pre-LN  (actuel) : x = x + SubLayer(LayerNorm(x))  ← plus stable

        normed = self.layernorm1(enc_input)
        attn_out, alpha = self.mha(
            normed, normed, normed,
            key_padding_mask=mask_padding,
            need_weights=need_weights,
        )
        x = enc_input + self.dropout(attn_out)

        normed   = self.layernorm2(x)
        ff_out   = self.ffn(normed)
        x        = x + self.dropout(ff_out)

        if need_weights:
            return x, alpha

        return x