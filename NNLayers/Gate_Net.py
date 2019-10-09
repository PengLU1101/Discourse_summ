#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-08-21
from typing import List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
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

        if score_type == 'bilinear':
            self.func: Callable[[T, T], T] = nn.Bilinear(dim_in, dim_in, 1)
        else:
            self.func: Callable[[T, T], T] = torch.bmm
        self.init_para()

    def forward(self, rep_srcs: List[T]) -> T:
        """
        :param rep_srcs:
        :return:
        """
        #assert rep_srcs.size(0)
        rep_with_head = torch.cat(
            tuple(map(self.cat_h, rep_srcs)), dim=0
        )[:, None, :] # (B x seq) x 1 x dim_hid
        rep_with_tail = torch.cat(
            tuple(map(self.cat_t, rep_srcs)),
            dim=0
        )
        if self.score_type == 'bilinear':
            score = self.func(
                rep_with_head,
                self.Dropout(rep_with_tail[:, None, :])
            ) # (B x seq) x 1 x 1
        else:
            score = self.func(
                rep_with_head,
                self.Dropout(rep_with_tail[:, :, None])
            )
        return score.squeeze(-1).squeeze(-1) # (B x seq)

    def parsing(self,
                rep_srcs: List[T],
                rep_tgts: List[T]):
        """
        :param rep_srcs:
        :param rep_tgts:
        :return:
        """
        assert len(rep_srcs) == len(rep_tgts)
        pass


    def parsing_score(self, src: T, tgt: T):
        assert src.size(1) == tgt.size(1)
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
        nn.init.uniform_(
            tensor=self.head,
            a=-scope,
            b=scope
        )

        nn.init.uniform_(
            tensor=self.tail,
            a=-scope,
            b=scope
        )
        if self.score_type == 'bilinear':
            nn.init.xavier_normal_(
                self.func.weight
            )
            nn.init.zeros_(
                self.func.bias
            )


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
            score_by_doc.append(
                torch.index_select(score,
                                   0,
                                   torch.LongTensor(idx_list).to(score.device)
                )
            )
        gate_list: List[Tuple[T, T]] = []
        for score in score_by_doc:
            gate_list.append(self.compute_gate(score))
        return gate_list

    def pad_score(self,
                  score: T) -> T:
        pad_score = torch.cat(
            [torch.zeros(score.size(0)).to(score.device), score],
            dim=0
        )
        return torch.stack(
            [pad_score[i: i + score.size()[0]] for i in range(score.size()[0] - 1, 0, -1)],
            dim=0
        )

    def compute_prob(self,
                     score_hat: T,
                     score: T) -> T:
        if self.hard:
            gate = (F.hardtanh(
                (score_hat - score[None, :]) / self.resolution * 2 + 1
            ) + 1) / 2
        else:
            gate = F.sigmoid(
                (score_hat - score[None, :]) / self.resolution * 10 + 5
            )
        return gate

    def compute_gate(self,
                     score: T) -> Tuple[T, T]:
        #assert score.size()[0] > 4
        score = score[1: -1]
        fwd_gate = self.compute_prob(
            self.pad_score(score),
            score
        )
        bwd_gate = self.compute_prob(
            self.pad_score(torch.flip(score, dims=(0,))),
            torch.flip(score, dims=(0,))
        )
        fwd_gate = torch.cumprod(fwd_gate, dim=0)  # seq x seq - 1
        bwd_gate = torch.cumprod(bwd_gate, dim=0)  # seq x seq - 1
        return (fwd_gate, bwd_gate)

    def cpt_gate(self,
                     semantic_score: T) -> Tuple[T, T]:
        assert semantic_score.size()[0] > 4
        score = semantic_score[1 : -1] # (num_score - 2)
        fwd_score = torch.cat(
            [torch.zeros(score.size(0)), score],
            dim=0
        )
        bwd_score = torch.cat(
            [score, torch.zeros(score.size(0))],
            dim=0
        )
        fwd_score_hat = torch.stack(
            [fwd_score[i: i + score.size()[0]] for i in range(score.size()[0] - 1, 0, -1)],
            dim=0
        )
        bwd_score_hat = torch.stack(
            [bwd_score[i: i + score.size(0)] for i in range(1, score.size()[0])],
            dim=0
        )
        if self.hard:
            fwd_gate = (F.hardtanh(
                (fwd_score_hat - score[None, :]) / self.resolution * 2 + 1
            ) + 1) / 2
            bwd_gate = (F.hardtanh(
                (bwd_score_hat - score[None, :]) / self.resolution * 2 + 1
            ) + 1) / 2
        else:
            fwd_gate = F.sigmoid(
                (fwd_score_hat - score[None, :]) / self.resolution * 10 + 5
            )
            bwd_gate = F.sigmoid(
                (bwd_score_hat - score[None, :]) / self.resolution * 10 + 5
            )
        fwd_gate = torch.cumprod(fwd_gate, dim=0) # seq x seq - 1
        bwd_gate = torch.cumprod(bwd_gate, dim=0) # seq x seq - 1
        return (fwd_gate, bwd_gate)


def test():
    def get_idx_by_lens(lens_list):
        idx_list: List[List[int]] = []
        start = 0
        for i in range(len(lens_list)):
            idx_list += [list(range(start, start + lens_list[i]))]
            start = idx_list[-1][-1] + 1
        return idx_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_model = Score_Net(10, 0.5, 'dot').to(device)
    gate_model = Gate_Net(10, 0.5, 1, True).to(device)

    test_rep: List[T] = []
    rep_list: List[List[int]] = []
    score_list: List[List[int]] = []
    len_list: List[int] = []
    for i in range(5, 7):
        test_rep.append(torch.Tensor(i, 10).uniform_(-1, 1).cuda())
        len_list.append(i)
    rep_list = get_idx_by_lens(len_list)
    score_list = get_idx_by_lens([x + 1 for x in len_list])
    score = score_model(test_rep)
    gate1 = gate_model(score, test_rep, rep_list, score_list)
    print(gate1)

if __name__ == "__main__":
    test()
