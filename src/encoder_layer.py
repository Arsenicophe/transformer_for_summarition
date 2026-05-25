from torch import nn 
import torch


class EncoderLayer (nn.Module) :

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha       = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        self.ffn = nn.Linear(d_model, d_model)


    def forward(self, enc_input, mask_padding, need_weights = False):

        x, alpha = self.mha(
            enc_input, enc_input, enc_input,
            key_padding_mask=mask_padding,
            need_weights=need_weights,
        )
        # Add & Norm
        x = self.layernorm1(enc_input + x)
        out_ff  = self.ffn(x)
        # Add & Norm
        x = self.layernorm2( x + self.dropout(out_ff))
        if need_weights :
            return x, alpha
        return x
    
