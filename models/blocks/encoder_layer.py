#%%
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


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        # TODO
        self.Feed_Forward = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.attention = MultiHeadAttention(d_model, n_head)
        self.Norm1 = LayerNorm(d_model)
        self.Norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        
        
    def forward(self, x, src_mask):
        # TODO
        x_old = x
        x = self.attention(x, x, x, src_mask)
        x = x_old+x
        x = self.Norm1(x)
        x = self.dropout1(x)

        x_old = x
        x = self.Feed_Forward(x)
        x = x_old+x
        x = self.Norm2(x)
        x = self.dropout2(x)

        return x

# %%
