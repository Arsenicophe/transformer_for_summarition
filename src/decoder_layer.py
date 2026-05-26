from torch import nn
import torch


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.mha1 = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.mha2 = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, query, context, mask_padding=None, causal_mask=None):

        out1, _ = self.mha1(
            query, query, query,
            attn_mask=causal_mask
        )

        query = self.layernorm1(
            query + self.dropout(out1)
        )

        out2, _ = self.mha2(
            query, context, context,
            key_padding_mask=mask_padding
        )

        out2 = self.layernorm2(
            query + self.dropout(out2)
        )

        out_ffn = self.ffn(out2)

        logits = self.layernorm3(
            out2 + self.dropout(out_ffn)
        )

        return logits