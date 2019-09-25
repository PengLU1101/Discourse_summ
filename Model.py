#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-07-16
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

from NNLayers.Embeddings import Embeddings
from NNLayers.Transformer import TransformerEncoder
from NNLayers.ParsingNN import Parsing_Net

class PEmodel(nn.module):
    def __init__(self,
                 embedding_layer,
                 encoder,
                 parser,
                 predictor):
        super(PEmodel, self).__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.parser = parser
        self.predictor = predictor

    def forward(self):
        pass

    @staticmethod
    def train_step(model,
                   optimizer,
                   train_iterator,
                   args):
        model.train()
        optimizer.zero_grad()

        # ________ = next(train_iterator)
        pass

    @staticmethod
    def test_step(model,
                  test_iterator,
                  args):
        model.eval()
        pass




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