#%%
"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        # TODO
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model(512)]
        :return: [batch_size, head, length, d_tensor]
        """
        # TODO
        assert tensor.shape[2] % self.n_head == 0
        tensor = tensor.view(tensor.shape[0], tensor.shape[1], self.n_head, tensor.shape[2]//self.n_head).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        # TODO
        tensor = tensor.transpose(1,2)
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2]*tensor.shape[3])
        return tensor


    def forward(self, q, k, v, mask=None):
        # TODO
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, _ = self.attention(q, k, v, mask)  # out: [batch_size, head(8), length, d_tensor(64)], score: [batch_size, head(8), length, length]
        out = self.concat(out)
        out = self.w_o(out)
        
        return out


# %%
