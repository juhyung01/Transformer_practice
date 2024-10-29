#%%
"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import torch

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # TODO
        self.Linear1 = nn.Linear(in_features=d_model, out_features=hidden)
        self.Linear2 = nn.Linear(in_features=hidden, out_features=d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        # TODO
        x = self.Linear2(self.dropout(torch.relu(self.Linear1(x))))
        return x

# %%
