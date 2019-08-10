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
                 encoder_layer,
                 parsing_layer,
                 infer_layer):
        super(PEmodel, self).__init__()
        self.embedding_layer = embedding_layer
        self.encoder_layer = encoder_layer
        self.parsing_layer = parsing_layer
        self.infer_layer = infer_layer

    def forward(self):
        pass


def build_model(para):
    assert isinstance(para, namedtuple)

    return PEmodel(embedding_layer, encoder_layer, parsing_layer, infer_layer)


def test():
    pass


if __name__ = "__main__":
    test()