"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import os
os.chdir('..')

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        # TODO
        self.Masked_MHA = MultiHeadAttention(d_model, n_head)
        self.Norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.MHA = MultiHeadAttention(d_model, n_head)
        self.Norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.FeedForward = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.Norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)


    def forward(self, dec, enc, trg_mask, src_mask):
        # TODO
        x = dec
        x_old = dec
        x = self.Masked_MHA(x,x,x,trg_mask)
        x = self.dropout1(x)
        x = self.Norm1(x + x_old)

        x_old = x
        x = self.MHA(x, enc, enc, src_mask)
        x = self.dropout2(x)
        x = self.Norm2(x + x_old)

        x_old = x
        x = self.FeedForward(x)
        x = self.dropout3(x)
        x = self.Norm3(x + x_old)

        return x
