from encoder_layer import Decoder_layer
import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, n_layer, n_heads, dropout):
        super(Decoder).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = [Decoder_layer(embedding_dim, n_heads, dropout) 
                       for _ in range(n_layer)]
        

    def forward():
