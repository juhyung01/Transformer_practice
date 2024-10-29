#%%
"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12): # [batch_size, query_len]
        # TODO
        d = k.shape[-1]
        score = q@k.transpose(-1, -2)/math.sqrt(d)      # MatMul, scale, [batch_size, batch_size]
        if mask == None: pass                           # mask(있을 때만)
        else:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.softmax(score)
        out = score@v                                   # MatMul, [batch_size, query_len]
        return out, score

# %%
