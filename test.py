#%%
import math
import torch

from torch import nn, optim
from torch.optim import Adam

#from data import *
from models.model.transformer import Transformer

model = Transformer(src_pad_idx=2,
                    trg_pad_idx=2,
                    trg_sos_idx=3,
                    d_model=512,
                    enc_voc_size=30,
                    dec_voc_size=30,
                    max_len=30,
                    ffn_hidden=2048,
                    n_head=8,
                    n_layers=6,
                    drop_prob=0.1,
                    device='cpu')
# %%
