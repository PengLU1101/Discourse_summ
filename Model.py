#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-07-16
from typing import List, Tuple, Dict, Union, Callable
from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
import numpy as np

INF = 1e-9
from NNLayers.Embeddings import Embedding_Net, WordEmbedding, PositionalEncoding
from NNLayers.Gate_Net import Gate_Net, Score_Net
from NNLayers.Predict_Net import Predic_Net


class Encoder(nn.Module):
    def __init__(self,
                 emb_layer,
                 enc_layer):
        super(Encoder, self).__init__()
        self.Emb_Layer = emb_layer
        self.Enc_Layer = enc_layer
        if emb_layer.dim != enc_layer.d_model:
            self.prejector = nn.Linear(emb_layer.dim, enc_layer.d_model)

    def forward(self,
                src: T,
                mask: T,
                idx_list: List[List[int]]) -> List[T]:
        rep = self.Emb_Layer(src).permute(1, 0, 2)
        rep = self.Enc_Layer(
            src=rep,
            src_key_padding_mask=mask.eq(0)).permute(1, 0, 2)[:, 0, :]
        if self.emb_layer.dim != self.enc_layer.d_model:
            rep = self.prejector(rep)
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
                   data_iterator,
                   args):
        model.train()
        optimizer.zero_grad()

        Tensor_dict, token_dict, idx_dict = next(data_iterator)
        pos_loss, neg_loss = model(
            Tensor_dict['src'],
            Tensor_dict['mask_src'],
            idx_dict['rep_idx'],
            idx_dict['score_idx']
        )
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
                  data_iterator,
                  args):
        model.eval()
        Tensor_dict, token_dict, idx_dict = next(data_iterator)
        pos_loss, neg_loss = model(
            Tensor_dict['src'],
            Tensor_dict['mask_src'],
            idx_dict['rep_idx'],
            idx_dict['score_idx']
        )
        loss = (pos_loss - neg_loss) / 2

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

def build_model(para):

    word_emb = WordEmbedding(para.word2id, para.emb_dim)
    position_emb = PositionalEncoding(para.dropout, para.emb_dim)
    emb_layer = Embedding_Net(
        word_emb,
        position_emb,
    )
    enc_layer = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(para.emb_dim, para.nhead, para.dropout),
        num_layers=para.n_layer,
        norm=nn.LayerNorm(para.d_model)
    )
    encoder = Encoder(emb_layer, enc_layer)

    score_layer = Score_Net(
        para.d_model,
        para.dropout,
        para.score_type
    )
    gate_layer = Gate_Net(
        para.d_model,
        para.dropout,
        para.resolution,
        para.hard
    )
    parser = Parser(score_layer, gate_layer)

    predictor = Predic_Net(
        para.d_model,
        para.score_type
    )

    return PEmodel(encoder, parser, predictor)

def get_idx_by_lens(lens_list: List[int]) -> List[List[int]]:
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
        _ += [random.choice(ll[i + 1: i + 6])]
    return _

def test():
    word_emb = WordEmbedding(100, 20)
    position_emb = PositionalEncoding(0.5, 20)
    emb_layer = Embedding_Net(
        word_emb,
        position_emb,
    )
    enc_layer = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=20, nhead=2, dropout=0.5),
        num_layers=3,
        norm=nn.LayerNorm(20)
    )
    encoder = Encoder(emb_layer, enc_layer)

    score_layer = Score_Net(
        20,
        0.5,
        'dot'
    )
    gate_layer = Gate_Net(
        20,
        0.5,
        1,
        'dot'
    )
    parser = Parser(score_layer, gate_layer)

    predictor = Predic_Net(
        20,
        'dot'
    )
    model = PEmodel(encoder, parser, predictor)

    data = list(range(100))
    random.shuffle(data)
    t = torch.LongTensor(data).view(25, 4)
    mask = torch.ones(25, 4).long()
    #mask[0, :] = 0
    len_list = [4, 5, 10, 6]
    repidx = get_idx_by_lens(len_list)
    scoreidx = get_idx_by_lens([x+1 for x in len_list])
    neg_idx = get_neglist(len_list)

    lp, ln = model(t, mask, repidx, scoreidx, neg_idx)
    print(lp)
    print(ln)

if __name__ == "__main__":
    test()