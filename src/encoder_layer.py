from torch import nn

class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, enc_input, mask_padding=None, need_weights=False):

        attn_out, alpha = self.mha(
            enc_input,
            enc_input,
            enc_input,
            key_padding_mask=mask_padding,
            need_weights=need_weights,
        )

        x = self.layernorm1(
            enc_input + self.dropout(attn_out)
        )

        ff_out = self.ffn(x)

        x = self.layernorm2(
            x + self.dropout(ff_out)
        )

        if need_weights:
            return x, alpha

        return x