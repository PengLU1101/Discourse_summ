#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-09-21

from typing import List, Tuple, Dict, Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T


import numpy as np

StateType = Dict[str, T]  # pylint: disable=invalid-name
StepFunctionType = Callable[[T, StateType], Tuple[T, StateType]]  # pylint: disable=invalid-name

class Predic_Net(nn.Module):
    def __init__(self,
                 dim_hid: int,
                 score_type: str,
                 bidirectional: bool = False) -> None:
        super(Predic_Net, self).__init__()
        self.dim_hid = dim_hid
        self.score_type = score_type
        self.bidirectional = bidirectional
        if score_type == 'bilinear':
            self.func: Callable[[T, T], T] = nn.Bilinear(dim_hid, dim_hid, 1)
        elif score_type == 'dot':
            self.func: Callable[[T, T], T] = torch.bmm
        elif score_type == 'denselinear':
            self.func = nn.Linear(4 * dim_hid, 2)
        self.init_para()
        self.norm_factor = np.sqrt(dim_hid)

    def forward(self,
                rep_sents: List[T],
                gate: List[Tuple[T, T]],
                fwd_neg: T,
                bwd_neg: T) -> Dict[str, T]:
        fwd: List[T] = list(map(
            self.get_sm, 
            rep_sents
            )) #item h1 h2 h3 h4
        bwd: List[T] = list(map(
            self.get_sm, 
            [x.flip((0, )) for x in rep_sents]
            )) #item h3 h2 h1 h0
        mask: List[Tuple[T, T]] = list(map(
            self.mask_gate,
            gate
            ))
        doc_fwd, doc_bwd = self.compute_h(fwd, bwd, mask)

        fwd_h = torch.cat(doc_fwd, dim=0)
        fwd_pos = torch.cat(
            [rep[1:, :] for rep in rep_sents],
            dim=0
        )
        #fwd_neg = torch.cat(fwd_neg, dim=0)
        bwd_h = torch.cat(doc_bwd, dim=0)
        bwd_pos = torch.cat(
            [rep.flip((0,))[1:, :] for rep in rep_sents],
            dim=0
        )
        #bwd_neg = torch.cat(bwd_neg, dim=0) ##### ORDER MATTERS!!!!!!!!!!!!!!!!!!!!!11
        if self.score_type == 'denselinear':
            fp_lld = F.log_softmax(self.cpt_logit(fwd_h, fwd_pos), dim=-1)
            fn_lld = F.log_softmax(self.cpt_logit(fwd_h, fwd_neg), dim=-1)
            if self.bidirectional:
                bp_lld = F.log_softmax(self.cpt_logit(bwd_h, bwd_pos), dim=-1)
                bn_lld = F.log_softmax(self.cpt_logit(bwd_h, bwd_neg), dim=-1)
        else:
            fp_lld = F.logsigmoid(self.cpt_logit(fwd_h, fwd_pos))
            fn_lld = F.logsigmoid(-self.cpt_logit(fwd_h, fwd_neg))
            if self.bidirectional:
                2
            pos_loss = torch.mean(fp_lld)
            neg_loss = torch.mean(fn_lld)
        loss = {'fwd_pos': fwd_pos, 'fwd_neg': fwd_neg}
        if

        return (pos_loss, neg_loss)

    def cpt_logit(self, h: T, t: T) -> T:
        if self.score_type == 'bilinear':
            lld = self.func(h[:, None, :], t[:, None, :])  # BxNx1
            lld = lld.squeeze(-1) / self.norm_factor
        elif self.score_type == 'dot':
            lld = self.func(h[:, None, :], t[:, :, None])
            lld = lld.squeeze(-1) / self.norm_factor
        elif self.score_type == 'denselinear':
            h = h[:, None, :]
            t = t[:, None, :],
            lld = self.func(torch.cat(
                h, t, h*t, torch.abs(h-t)
            ))

        return lld



    def compute_h(self,
                  fwd: List[T],
                  bwd: List[T],
                  mask: List[Tuple[T,T]]) -> Tuple[List[T], List[T]]:
        fwd_list: List[T] = []
        bwd_list: List[T] = []

        for i, square in enumerate(zip(fwd, bwd)):
            doc_fwd = torch.sum(
                square[0] * mask[i][0][:, :, None],
                dim=0
            ) * torch.sum(mask[i][0], dim=0)[:, None]  # h1, h2, h3, h4 (N x dim)
            doc_bwd = torch.sum(
                square[1] * mask[i][1][:, :, None],
                dim=0
            ) * torch.sum(mask[i][1], dim=0)[:, None]  # h3, h2, h1, h0
            fwd_list.append(doc_fwd)
            bwd_list.append(doc_bwd)
        return (fwd_list, bwd_list)



    def get_sm(self, rep: T) -> T:
        pad_rep = torch.cat(
            [torch.zeros(rep.size(0) - 2, rep.size(1)).to(rep.device), rep[:-1, :]],
            dim=0
        )
        square = torch.stack(
            [pad_rep[i: i+rep.size(0)-1, :].flip((0,)) for i in range(0, rep.size(0)-1)],
            dim = 0
        )
        return square #

    def mask_gate(self, 
                  gate_tuple: Tuple[T, T]) -> Tuple[T, T]:
        """
        :param gate_tuple:
        :return: (N x N)
        """
        assert gate_tuple[0].size(0) + 1 == gate_tuple[0].size(1), \
            f"The expected dim of gate is (n-1, n)"
        assert gate_tuple[1].size(0) + 1 == gate_tuple[1].size(1), \
            f"The expected dim of gate is (n-1, n)"

        return (torch.triu(torch.cat((torch.ones(1, gate_tuple[0].size(1)).to(gate_tuple[0].device),
                                      gate_tuple[0]), 
                                      dim=0),  
                           diagonal=0), 
                torch.triu(torch.cat((torch.ones(1, gate_tuple[1].size(1)).to(gate_tuple[1].device),
                                      gate_tuple[1]), 
                                      dim=0), 
                           diagonal=0)
               )
    def init_para(self):
        if self.score_type == 'bilinear':
            nn.init.xavier_normal_(
                self.func.weight
            )
            nn.init.zeros_(
                self.func.bias
            )


def test():
    pass


if __name__ == "__main__":
    test()