#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-28


import senteval


def prepare():
    pass

def batcher():
    pass

params = {
    'task_path': PATH_TO_DATA,
    'usepytorch': True,
    'kfold': 10
}

params['classifier'] = {
    'nhid': 0,
    'optim': 'adam',
    'batch_size': 64,
    'tenacity': 5,
    'epoch_size': 4}
se = senteval.engine.SE(params, batcher, prepare)
transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                  'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                  'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                  'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                  'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
results = se.eval(transfer_tasks)

if __name__ == "__main__":
    test()