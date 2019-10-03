""" Embeddings module """
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor as T
import numpy as np



class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self,
                 dropout: float,
                 dim: int,
                 max_len: int=5000) -> None:
        if dim % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={dim :d})")
        pe = torch.zeros(max_len, dim) # max_len x dim
        position = torch.arange(0, max_len).unsqueeze(1) # max_len x 1
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0) # 1 x max_len x dim
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self,
                emb: T,
                step: Optional[int]=None) -> T:
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:, :emb.size(1), :]
        else:
            emb = emb + self.pe[:, step, :]
        return self.dropout(emb)


class WordEmbedding(nn.Module):
    def __init__(self,
                 vocab: int,
                 dim: int):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, dim)
        self.dim = dim
        scope = np.sqrt(1.0 / dim)
        self.lut.weight.data.uniform_(-scope, scope)
        self.lut.weight.data[0] = torch.zeros(1, dim)

    def forward(self, x: T) -> T:
        """
        Arguments:
            x: [batch_size, seq_len] LongTensor
        Output:
            embeds: [batch, seq_len, d_emb] FloatTensor
        """
        embeds = self.lut(x)

        return embeds
    def apply_weights(self, weights, fine_tune_flag=True):
        if isinstance(weights, np.ndarray):
            self.lut.weight.data.copy_(torch.from_numpy(weights))
        else:
            pass
        if not fine_tune_flag:
            for p in self.lut.parameters():
                p.requires_grad = False

class Embedding_Net(nn.Module):
    def __init__(self, word_emb, position_emb):
        super(Embedding_Net, self).__init__()
        self.wordemb = word_emb
        self.positionemb = position_emb

    def forward(self, input: T) -> T:
        return self.positionemb(self.wordemb(input))