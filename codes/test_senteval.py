#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-28

import logging
import sys
sys.setdefaultencoding('utf8')
import senteval
from Model import PEmodel
import os, random
from typing import List, Dict
from itertools import chain
from collections import defaultdict, OrderedDict
import json
import re
import pickle

import torch
import torch.utils.data as data

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_SKIPTHOUGHT = ''


def collate_fn(data):
    def pad_mask(data):
        chain_src = list(chain.from_iterable([_ for _ in data]))
        src_lens = [len(_) for _ in chain_src]
        max_src_lens = max(src_lens)
        padded_src = torch.zeros(len(chain_src), max_src_lens).long()
        mask_src = torch.zeros(len(chain_src), max_src_lens).long()
        for i, sent in enumerate(chain_src):
            end = src_lens[i]
            padded_src[i, :end] = torch.LongTensor(sent[:end])
            mask_src[i, :end] = 1
        return padded_src, mask_src, src_lens
    padded_src, mask_src, src_lens = pad_mask(data_idx)
    Tensor_dict = {'src': padded_src,
                   'mask_src': mask_src,
                   }
    length_dict = {'src': src_lens}
    return Tensor_dict, length_dict
def prepare():
    pass


def batcher(params, batch):
    sentences = [[params['word2id'][x] for x in ['<start>'] + s + ['<end>']] for s in batch]
    Tensor_dict, length_dict = collate_fn(sentences)

    embeddings = PEmodel.encode(
        params['encoder'],
        Tensor_dict['src'],
        Tensor_dict['mask_src'],
        length_dict
    )
    return embeddings

params_senteval = {
    'task_path': PATH_TO_DATA,
    'usepytorch': True,
    'kfold': 10
}

params_senteval['classifier'] = {
    'nhid': 0,
    'optim': 'adam',
    'batch_size': 64,
    'tenacity': 5,
    'epoch_size': 4
}


if __name__ == "__main__":
    params_senteval['word2id'] = load_pkl(path_word2id)
    params_senteval['encoder'] = PEmodel.load_model()
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                      'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
