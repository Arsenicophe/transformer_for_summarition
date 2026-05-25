from torch import nn
import torch


class Decoder_layer(nn.Module) :

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha1       = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.mha2       = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        self.ffn = nn.Linear(d_model, d_model)

    def forward(self, qerry, context,mask_padding, causal_mask) :

        out1 = self.mha1(
            qerry,qerry,qerry,
            attn_mask=causal_mask)
        
        qerry = self.layernorm1(out1 + qerry)

        out2 = self.mha2(
            qerry,context,context, 
            key_padding_mask=mask_padding)
        
        out2 = self.layernorm2(out2 + qerry)

        out_ffn = self.ffn(out2)

        logit = self.layernorm3(out2 + self.dropout(out_ffn))

        return logit