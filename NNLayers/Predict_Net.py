#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-09-21

from typing import List, Tuple, Callable

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
                 func: str) -> None:
        super(Predic_Net, self).__init__()
        self.dim_hid = dim_hid
        self.func = func
        if func == 'dot':
            pass
        else:
            self.func = pass

    def forward(self,
                rep_sents: List[T],
                gate: List[Tuple[T]],
                idx: List[List[int]],
                pos: List[List[int]],
                neg: List[List[int]]) -> T:
        fwd_list: List[T] = map(self.get_sm, rep_sents)
        bwd_list: List[T] =



    def get_sm(self, rep: [T]) -> T:
        pad_rep = torch.cat([torch.zeros(rep.size(0) - 1, rep.size(1)).to(rep.device), rep], dim=0)
        square = torch.stack(
            [rep[i: i+rep.size(0)].flip((0,)) for i in range(0, rep.size(0))],
            dim = 0
        )
        return square

    def mask_gate(self, gate: T) -> T:
        assert gate.size(0) < gate.size(1)
        return torch.triu(
            torch.cat((torch.ones(1, gate.size(1)), gate), dim=0),
            diagonal=0
        )


def test():
    pass


if __name__ == "__main__":
    test()