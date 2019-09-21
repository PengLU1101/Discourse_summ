import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T

from typing import List, Tuple

import numpy as np

class Score_Net(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dropout: float,
                 score_type : str ='dot') -> None:
        super(Score_Net, self).__init__()
        self.dim_in = dim_in
        self.dropout = dropout
        self.score_type = score_type
        self.Dropout = nn.Dropout(dropout)

        if score_type == 'bilinear':
            self.score_func = nn.Bilinear(dim_in, dim_in, 1)
        elif score_type == 'dot':
            self.score_func = torch.bmm

        self.init_para()

    def forward(self, rep_srcs: List[T]) -> T:
        #assert rep_srcs.size(0)
        rep_with_head = torch.cat(tuple(map(self.cat_h, rep_srcs)), dim=0)[:, None, :] # (B x seq) x 1 x dim_hid
        rep_with_tail = torch.cat(tuple(map(self.cat_t, rep_srcs)), dim=0)[:, None, :]
        score = self.score_func(rep_with_head, self.Dropout(rep_with_tail)) # (B x seq) x 1 x 1

        return score.squeeze(-1).squeeze(-1) # (B x seq)

    def parsing(self,
              rep_srcs: List[T],
              rep_tgts: T) -> T:
        pass

    def cat_h(self, rep: T) -> T:
        """
        Cat a head rep. to the rep. of sents of a doc. To get a Tensor: [head, s1, s2, s3]
        :param rep: num_sent x dim_hid
        :return: rep_with_head: (num_sent + 1) x dim_hid
        """
        return torch.cat([self.head, rep], dim=0)
    def cat_t(self, rep: T) -> T:
        """
        Cat a tail rep. to the rep. of sents of a doc. To get a Tensor: [s1, s2, s3, tail]
        :param rep: num_sent x dim_hid
        :return: rep_with_head: (num_sent + 1) x dim_hid
        """
        return torch.cat([rep, self.tail], dim=0)

    def init_para(self):
        scope = np.sqrt(1.0 / self.dim_in)
        nn.init.uniform(tensor=self.head,
                        a=-scope,
                        b=scope)

        nn.init.uniform(tensor=self.tail,
                        a=-scope,
                        b=scope)
        if self.score_type == 'bilinear':
            nn.init.xavier_normal_(self.score_func.weight)
            nn.init.zeros_(self.score_func.bias)

class Gate_Net(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dropout: float,
                 resolution: float,
                 hard: bool) -> None:
        super(Gate_Net, self).__init__()
        self.dim = dim_in
        self.dropout = dropout
        self.resolution = resolution
        self.hard = hard
        self.Dropout = nn.Dropout(dropout)

    def forward(self,
                score: T,
                rep_srcs: List[T],
                rep_idx: List[List[int]]) -> List[Tuple[T]]:
        score_by_doc = []
        for idx in score_idx:
            score_by_doc += [torch.index_select(input=score,
                                                dim=0,
                                                index=torch.LongTensor(idx))]
        gate_list = []
        for score in score_by_doc:
            gate_list.append(self.compute_gate(score))

        return gate_list



    def compute_gate(self,
                     semantic_score: T) -> Tuple[T]:
        assert semantic_score.size(0) > 4
        fwd_score = semantic_score[: -1] # (num_score - 2)
        bwd_score = semantic_score[1: ]
        pad_fwd_score = torch.cat([torch.zeros(fwd_score.size(0) - 1), fwd_score], dim=0)
        pad_bwd_score = torch.cat([bwd_score, torch.zeros(bwd_score.size(0) - 1)], dim=0)
        fwd_score_hat = torch.stack([pad_fwd_score[i: i + pad_score.size(0)] for i in range(pad_score.size(0) - 1, 0, -1)], dim=1)
        bwd_score_hat = torch.stack([pad_bwd_score[i: i + pad_score.size(0)] for i in range(1, pad_score.size(0))], dim=1)
        if self.hard:
            fwd_gate = (F.hardtanh((score_hat - fwd_score[:, None]) / self.resolution * 2 + 1) + 1) / 2
            bwd_gate = (F.hardtanh((score_hat - bwd_score[:, None]) / self.resolution * 2 + 1) + 1) / 2

        else:
            fwd_gate = F.sigmoid((score_hat - fwd_score[:, None]) / self.resolution * 10 + 5)
            bwd_gate = F.sigmoid((score_hat - bwd_score[:, None]) / self.resolution * 10 + 5)
        fwd_gate = torch.cumprod(fwd_gate, dim=1) # seq x seq - 1
        bwd_gate = torch.cumprod(bwd_gate, dim=1) # seq x seq - 1

        return (fwd_gate, bwd_gate)


def test():
    pmodel = Parsing_Net(dim_in=100,
                         dim_hid=64,
                         n_slots=5,
                         n_lookback=1,
                         resolution=0.1,
                         dropout=0.5)

    a, b = pmodel.init_hidden(16)


if __name__ == "__main__":
    test()
