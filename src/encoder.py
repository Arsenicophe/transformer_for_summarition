from encoder_layer import EncoderLayer
import torch.nn as nn
from utils import encodage_positionnel


class Encoder(nn.Module):

    def __init__(
        self,
        embedding_dim,
        vocab_size,
        n_layer,
        n_heads,
        len_seq_encoding_max,
        dropout
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    embedding_dim,
                    n_heads,
                    dropout
                )
                for _ in range(n_layer)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "pos_encoding",
            encodage_positionnel(
                embedding_dim,
                len_seq_encoding_max
            )
        )

    def forward(
        self,
        enc_idx,
        mask_padding,
        need_weights=False
    ):

        seq_len = enc_idx.shape[1]

        pos_enc = self.pos_encoding[:, :seq_len, :]

        x = self.embedding(enc_idx)
        x = self.dropout(x + pos_enc)

        alpha = None

        for layer in self.layers:

            if need_weights:
                x, alpha = layer(
                    x,
                    mask_padding,
                    need_weights=True
                )
            else:
                x = layer(
                    x,
                    mask_padding,
                    need_weights=False
                )

        if need_weights:
            return x, alpha

        return x