#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-08-21

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T

from typing import List, Tuple, Dict, Union, Callable

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
        self.head = nn.Parameter(T(1, dim_in))
        self.tail = nn.Parameter(T(1, dim_in))
        self.func : Dict[str, Callable[[T, T], T]] = {'bilinear': nn.Bilinear(dim_in, dim_in, 1),
                                                      'dot': torch.bmm}
        self.init_para()

    def forward(self, rep_srcs: List[T]) -> T:
        #assert rep_srcs.size(0)
        print(f"self.head: {self.head.size()}")
        print(f"self.tail: {self.tail.size()}")
        #print(f"self.func.weight: {self.func[self.score_type].weight}")
        #print(f"self.func.bias: {self.func[self.score_type].bias}")

        rep_with_head = torch.cat(tuple(map(self.cat_h, rep_srcs)), dim=0)[:, None, :] # (B x seq) x 1 x dim_hid
        rep_with_tail = torch.cat(tuple(map(self.cat_t, rep_srcs)), dim=0)
        if self.score_type == 'bilinear':
            score = self.func[self.score_type](rep_with_head,
                                               self.Dropout(rep_with_tail[:, None, :])) # (B x seq) x 1 x 1
        else:
            score = self.func[self.score_type](rep_with_head,
                                               self.Dropout(rep_with_tail[:, :, None]))
        return score.squeeze(-1).squeeze(-1) # (B x seq)

    def parsing(self,
              rep_srcs: List[T],
              rep_tgts: T) -> T:
        pass

    def cat_h(self, rep: T) -> T:
        """
        Cat a head rep. to the rep. of sents of a doc.
        To get a Tensor: [head, s1, s2, s3]
        :param rep: num_sent x dim_hid
        :return: rep_with_head: (num_sent + 1) x dim_hid
        """
        return torch.cat([self.head, rep], dim=0)
    def cat_t(self, rep: T) -> T:
        """
        Cat a tail rep. to the rep. of sents of a doc.
        To get a Tensor: [s1, s2, s3, tail]
        :param rep: num_sent x dim_hid
        :return: rep_with_head: (num_sent + 1) x dim_hid
        """
        return torch.cat([rep, self.tail], dim=0)

    def init_para(self):
        scope = np.sqrt(1.0 / self.dim_in)
        nn.init.uniform_(tensor=self.head,
                        a=-scope,
                        b=scope)

        nn.init.uniform_(tensor=self.tail,
                        a=-scope,
                        b=scope)
        #if self.score_type == 'bilinear':
        #    nn.init.xavier_normal_(self.func[self.score_type].weight)
        #    nn.init.zeros_(self.func[self.score_type].bias)

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
                rep_idx: List[List[int]],
                score_idx: List[List[int]]) -> List[Tuple[T, T]]:
        score_by_doc: List[T] = []
        for idx_list in score_idx:
            score_by_doc += [torch.index_select(score, 0,
                                                torch.tensor(idx_list,
                                                             dtype=torch.int64))]
        gate_list: List[Tuple[T, T]] = []
        for score in score_by_doc:
            gate_list.append(self.compute_gate(score))

        return gate_list



    def compute_gate(self,
                     semantic_score: T) -> Tuple[T, T]:
        assert semantic_score.size()[0] > 4
        score = semantic_score[1 : -1] # (num_score - 2)
        fwd_score = torch.cat([torch.zeros(score.size(0)), score], dim=0)
        bwd_score = torch.cat([score, torch.zeros(score.size(0))], dim=0)
        fwd_score_hat = torch.stack([fwd_score[i: i + score.size()[0]]
                                     for i in range(score.size()[0] - 1, 0, -1)], dim=0)
        bwd_score_hat = torch.stack([bwd_score[i: i + score.size(0)]
                                     for i in range(1, score.size()[0])], dim=0)
        print(f"fwd_score_hat: {fwd_score_hat.size()}")
        if self.hard:
            fwd_gate = (F.hardtanh((fwd_score_hat - score[None, :])
                                   / self.resolution * 2 + 1) + 1) / 2
            bwd_gate = (F.hardtanh((bwd_score_hat - score[None, :])
                                   / self.resolution * 2 + 1) + 1) / 2

        else:
            fwd_gate = F.sigmoid((fwd_score_hat - score[None, :])
                                 / self.resolution * 10 + 5)
            bwd_gate = F.sigmoid((bwd_score_hat - score[None, :])
                                 / self.resolution * 10 + 5)

        fwd_gate = torch.cumprod(fwd_gate, dim=0) # seq x seq - 1
        bwd_gate = torch.cumprod(bwd_gate, dim=0) # seq x seq - 1
        print(f"fwd gate: {fwd_gate.data}")


        #print(f"bwd gate: {bwd_gate.data}")

        return (fwd_gate, bwd_gate)



def test():
    def get_idx_by_lens(lens_list):
        idx_list: List[List[int]] = []
        start = 0
        for i in range(len(lens_list)):
            idx_list += [list(range(start, start + lens_list[i]))]
            start = idx_list[-1][-1] + 1
        return idx_list
    score_model = Score_Net(10, 0.5, 'bilinear')
    gate_model = Gate_Net(10, 0.5, 1, True)

    test_rep: List[T] = []
    rep_list: List[List[int]] = []
    score_list: List[List[int]] = []
    len_list: List[int] = []
    for i in range(25, 26):
        test_rep.append(torch.Tensor(i, 10).uniform_(-1, 1))
        len_list.append(i)
    rep_list = get_idx_by_lens(len_list)
    score_list = get_idx_by_lens([x + 1 for x in len_list])
    score = score_model(test_rep)
    gate = gate_model(score, test_rep, rep_list, score_list)






    #print(score_model(test_rep))



if __name__ == "__main__":
    test()
