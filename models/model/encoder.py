"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import os
# os.chdir('..')

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # TODO
        self.n = n_layers
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, drop_prob, device)
        self.layer_encoder = EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)

    def forward(self, x, src_mask):
        # TODO
        x = self.embedding(x)
        for _ in range(self.n):
            x = self.layer_encoder(x, src_mask)
        return x