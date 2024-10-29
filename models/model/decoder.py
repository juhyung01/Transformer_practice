"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
import os
# os.chdir('..')

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # TODO
        self.n = n_layers
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)
        self.layer_decoder = DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
        self.linear = nn.Linear(d_model, dec_voc_size)
        self.softmax = nn.Softmax()

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # TODO
        dec = self.embedding(trg)
        for _ in range(self.n):
            output = self.layer_decoder(dec, enc_src, trg_mask, src_mask)
        output = self.linear(output)
        output = self.softmax(output)
        return output