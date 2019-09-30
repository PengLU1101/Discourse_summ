#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-07-16
from typing import List, Tuple, Dict, Union, Callable
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
import numpy as np

INF = 1e-9
from NNLayers.Embeddings import Embeddings
from NNLayers.Gate_Net import Gate_Net, Score_Net
from NNLayers.Predict_Net import Predic_Net

class Encoder(nn.Module):
    def __init__(self,
                 emb_layer,
                 enc_layer):
        super(Encoder, self).__init__()
        self.Emb_Layer = emb_layer
        self.Enc_Layer = enc_layer

    def forward(self,
                src: T,
                mask: T,
                idx_list: List[List[int]]) -> List[T]:
        rep = self.Enc_Layer(self.Emb_Layer(src).masked_fill_(mask[:, :, None].eq(0), INF))
        return [torch.index_select(rep,
                                  dim=0,
                                  index=torch.LongTensor(idx).to(src.device)) for idx in idx_list]

class Parser(nn.Module):
    def __init__(self,
                 score_layer,
                 gate_layer):
        super(Parser, self).__init__()
        self.score_layer = score_layer
        self.gate_layer = gate_layer
    def forward(self,
                rep_srcs: List[T],
                rep_idx: List[List[int]],
                score_idx: List[List[int]]) -> List[Tuple[T, T]]:
        scores = self.score_layer(rep_srcs)
        return self.gate_layer(scores, rep_srcs, rep_idx, score_idx)


class PEmodel(nn.Module):
    def __init__(self,
                 encoder,
                 parser,
                 predictor):
        super(PEmodel, self).__init__()
        self.encoder = encoder
        self.parser = parser
        self.predictor = predictor

    def forward(self,
                input: T,
                mask: T,
                rep_idx: List[List[int]],
                score_idx: List[List[int]],
                neg: List[Tuple[List[int], List[int]]]) -> Tuple[T, T]:
        reps: List[T] = self.encoder(input, mask, rep_idx)
        gate_list: List[Tuple[T, T]] = self.parser(reps, rep_idx, score_idx)
        return self.predictor(reps, gate_list, neg)############## Neg!!!!!!!!!!!!!!!!!!!!!!!

    @staticmethod
    def train_step(model,
                   optimizer,
                   train_iterator,
                   args):
        model.train()
        optimizer.zero_grad()

        Tensor_dict, token_dict, idx_dict = next(train_iterator)
        src = Tensor_dict['src']
        mask = Tensor_dict['mask_src']
        rep_idx = idx_dict['rep_idx']
        score_idx = idx_dict['score_idx']
        pos_loss, neg_loss = model(src, mask, rep_idx, score_idx)
        loss = (pos_loss - neg_loss) / 2

        loss.backward()
        optimizer.step()

        log = {
            #**regularization_log,
            'positive_sample_loss': pos_loss.item(),
            'negative_sample_loss': neg_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model,
                  test_iterator,
                  args):
        model.eval()
        ensor_dict, token_dict, idx_dict = next(test_iterator)
        src = Tensor_dict['src']
        mask = Tensor_dict['mask_src']
        rep_idx = idx_dict['rep_idx']
        score_idx = idx_dict['score_idx']
        pos_loss, neg_loss = model(src, mask, rep_idx, score_idx)
        loss = (pos_loss - neg_loss) / 2

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

def build_model(para):
    assert isinstance(para, namedtuple)
    embedding_layer = Embeddings(word_vec_size=para.emb_dim,
                                 word_vocab_size=para.voc_size,
                                 )

    return PEmodel(embedding_layer, encoder, parser, predictor)


def test():
    pass


if __name__ == "__main__":
    test()