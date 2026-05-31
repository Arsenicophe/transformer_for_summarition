from torch import nn


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.mha1 = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.mha2 = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, query, context, mask_padding=None, causal_mask=None):

        
        normed = self.layernorm1(query)
        out1, _ = self.mha1(
            normed, normed, normed,
            attn_mask=causal_mask,
        )
        query = query + self.dropout(out1)

        
        normed = self.layernorm2(query)
        out2, _ = self.mha2(
            normed, context, context,
            key_padding_mask=mask_padding,
        )
        query = query + self.dropout(out2)

        
        normed  = self.layernorm3(query)
        out_ffn = self.ffn(normed)
        output  = query + self.dropout(out_ffn)

        return output