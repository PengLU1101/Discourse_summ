#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-09-24
import unittest
import random

import torch

from NNLayers.ParsingNN import Score_Net, Gate_Net
from NNLayers.Predict_Net import Predic_Net


def get_idx_by_lens(lens_list):
    idx_list: List[List[int]] = []
    start = 0
    for i in range(len(lens_list)):
        idx_list += [list(range(start, start + lens_list[i]))]
        start = idx_list[-1][-1] + 1
    return idx_list
def get_neglist(lens_list):
    neg = []
    for x in lens_list:
        neg.append((smpneg(list(range(x))), smpneg(list(range(x))[::-1])))
    return neg

def smpneg(l):
    _ = []
    ll = l + l
    for i in range(1, len(l)):
        _ += [random.choice(ll[i+1: i+6])]
    return _

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_model = Score_Net(10, 0.5, 'bilinear').to(device)
    gate_model = Gate_Net(10, 0.5, 1, True).to(device)
    predict_model = Predic_Net(10, 'bilinear').to(device)

    test_rep: List[T] = []
    len_list: List[int] = []
    for i in range(5, 7):
        test_rep.append(torch.Tensor(i, 10).uniform_(-1, 1).cuda())
        len_list.append(i)
    rep_list = get_idx_by_lens(len_list)
    neg_list = get_neglist(len_list)
    score_list = get_idx_by_lens([x + 1 for x in len_list])
    score = score_model(test_rep)
    gate1 = gate_model(score, test_rep, rep_list, score_list)


    loss = predict_model(test_rep,
                         gate1,
                         neg=neg_list)


if __name__ == "__main__":
    test()