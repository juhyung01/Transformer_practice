#%%
"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.ones = nn.Parameter(torch.ones(d_model))  # grad_fn
        self.zeros = nn.Parameter(torch.zeros(d_model))  # grad_fn
        self.eps = eps
        self.d_model = d_model

    def forward(self, x):
        # TODO
        u = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x-u)/(std+self.eps)
        out = out*self.ones + self.zeros
        return out

# %%
