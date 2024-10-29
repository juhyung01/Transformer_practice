"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()
        # TODO
        self.d_model = d_model
        self.max_len = max_len
        self.encoding = torch.arange(0, max_len).float().unsqueeze(1).repeat(1, d_model)   # torch.cartesian_prod(torch.arange(0, max_len).float(), torch.arange(0, d_model).float())
        self.encoding[:,0::2] = torch.sin(self.encoding[:,0::2]/10000**(torch.arange(0, d_model, 2).float()/d_model))
        self.encoding[:,1::2] = torch.cos(self.encoding[:,1::2]/10000**(torch.arange(1, d_model, 2).float()/d_model))
        

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

