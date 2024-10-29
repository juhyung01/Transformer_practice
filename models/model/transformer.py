"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
import os
# os.chdir('..')
# os.chdir('transformer-master-skeleton\models')

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        # TODO
        self.src_pad_idx = src_pad_idx  # 인덱스 숫자
        self.trg_pad_idx = trg_pad_idx  # 인덱스 숫자
        self.trg_sos_idx = trg_sos_idx  # 인덱스 숫자
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)

    def make_src_mask(self, src):   # [batch_size, src_len]
        # TODO
        src_mask = (src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # src에서 src_pad_idx와 일치하는 항 빼고 전부 True 된다(원래 가릴 필요 없으니까)
        return src_mask # [batch_size, 1, 1, src_len]

    def make_trg_mask(self, trg):   # [batch_size, trg_len]
        # TODO
        trg_len = trg.shape[-1]
        trg_mask1 = (trg!=self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_mask2 = torch.ones(trg_len,trg_len)!=torch.tril(torch.ones(trg_len,trg_len))
        return trg_mask1 & trg_mask2    # [batch size, 1, trg len, trg len]

    def forward(self, src, trg):
        # TODO
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output