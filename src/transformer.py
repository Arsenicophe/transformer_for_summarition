from decoder import Decoder
from encoder import Encoder
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(
        self,
        embedding_dim,
        vocab_size,
        n_heads,
        n_layer,
        max_seq_encoding,
        dropout
    ):
        super().__init__()

        self.encoder = Encoder(
            embedding_dim,
            vocab_size,
            n_layer,
            n_heads,
            max_seq_encoding,
            dropout
        )

        self.decoder = Decoder(
            embedding_dim,
            vocab_size,
            n_heads,
            n_layer,
            max_seq_encoding,
            dropout
        )

    def forward(
        self,
        enc_idx,
        input_idx,
        enc_mask_padding,
        dec_mask_padding,
        causal_mask
    ):

        context = self.encoder(
            enc_idx,
            enc_mask_padding
        )

        logit = self.decoder(
            input_idx,
            context,
            dec_mask_padding,
            causal_mask
        )

        return logit