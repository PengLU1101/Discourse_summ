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

    def forward(self,
                src: T,
                mask: T,
                idx_list: List[List[int]]) -> List[T]:
        rep = self.Emb_Layer(src).permute(1, 0)
        rep = self.Enc_Layer(
            src=rep,
            src_key_padding_mask=mask.eq(0)).permute(1, 0)[:, 0, :]
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

    word_emb = WordEmbedding(para.voc_size, para.emb)
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
        para.d,
        para.dropout,
        para.score_type
    )
    gate_layer = Gate_Net(
        para.d,
        para.dropout,
        para.resolution,
        para.hard
    )
    parser = Parser(score_layer, gate_layer)

    predictor = Predic_Net(
        para.d,
        para.score_type
    )

    return PEmodel(encoder, parser, predictor)


def test():
    pass


if __name__ == "__main__":
    test()