from decoder_layer import DecoderLayer
import torch.nn as nn
from utils import encodage_positionnel


class Decoder(nn.Module):

    def __init__(
        self,
        embedding_dim,
        vocab_size,
        n_heads,
        n_layer,
        max_seq_encoding,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    embedding_dim,
                    n_heads,
                    dropout
                )
                for _ in range(n_layer)
            ]
        )

        self.register_buffer(
            "pos_encoding",
            encodage_positionnel(
                embedding_dim,
                max_seq_encoding
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_idx,
        context,
        padding_mask,
        causal_mask
    ):

        seq_len = input_idx.shape[1]

        pos_enc = self.pos_encoding[:, :seq_len, :]

        x = self.embedding(input_idx)

        x = self.dropout(x + pos_enc)

        for layer in self.layers:
            x = layer(
                x,
                context,
                padding_mask,
                causal_mask
            )

        return x