#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-09-21

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T

from typing import List, Tuple

import numpy as np


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
                rep_sents: T,
                gate: List[Tuple[T]],
                idx: List[List[int]],
                pos: List[List[int]],
                neg: List[List[int]]) -> T:
        pass



def test():
    pass


if __name__ == "__main__":
    test()