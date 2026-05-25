from encoder_layer import EncoderLayer
import torch.nn as nn
import torch
from utils import encodage_positionnel


class Encoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, n_layer, n_heads, len_seq_encoding_max, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = [EncoderLayer(embedding_dim, n_heads, dropout) 
                       for _ in range(n_layer)]
        self.dropout = nn.Dropout(dropout)
        self.n_layer = n_layer

        self.encodage_positionnel = encodage_positionnel(embedding_dim, len_seq_encoding_max)
                                          
    def forward(self, enc_idx , mask_padding , need_weights = False):
        seq_len = enc_idx.shape[0]

        pos_enc = self.encodage_positionnel[: seq_len]

        x = self.embedding(enc_idx) + pos_enc

        x = self.dropout(x)
        alpha = None 
        for layer  in self.layers :
            x, alpha = layer(x, mask_padding, need_weights)
        if need_weights :
            return x, alpha
        return x 
