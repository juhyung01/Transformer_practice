"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        # TODO
        self.TokenLayer = TokenEmbedding(vocab_size, d_model)
        self.drop_out = nn.Dropout(drop_prob)
        self.PositionLayer = PositionalEncoding(d_model, max_len, device)

    def forward(self, x):
        # TODO
        tok_emb = self.TokenLayer(x)
        pos_emb = self.PositionLayer(x)
        return self.drop_out(tok_emb + pos_emb)
