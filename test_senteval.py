#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-28

import logging
import sys
sys.setdefaultencoding('utf8')
import senteval

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_SKIPTHOUGHT = ''


def prepare():
    pass

def batcher(params, batch):

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
    params_senteval['encoder'] = PEmodel.load_model()
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                      'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
