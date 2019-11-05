#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-28

import logging
import sys
import os, random
from itertools import chain
import json
from collections import defaultdict

import torch
import numpy as np

# Set PATHs
PATH_TO_SENTEVAL = '/u/lupeng/Project/code/SentEval'
PATH_TO_DATA = '/u/lupeng/Project/code/SentEval/data/'

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
from Model import PEmodel, build_model
from Dataset import make_vocab
from Parser import read_pkl

def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id
def prepare(params, samples):
    return

def collate_fn(data):
    def pad_mask(data):
        src_lens = [len(_) for _ in data]
        max_src_lens = max(src_lens)
        padded_src = torch.zeros(len(data), max_src_lens).long()
        mask_src = torch.zeros(len(data), max_src_lens).long()
        for i, sent in enumerate(data):
            end = src_lens[i]
            padded_src[i, :end] = torch.LongTensor(sent[:end])
            mask_src[i, :end] = 1
        return padded_src, mask_src, src_lens
    padded_src, mask_src, src_lens = pad_mask(data)
    Tensor_dict = {'src': padded_src,
                   'mask_src': mask_src,
                   }
    length_dict = {'src': src_lens}
    return Tensor_dict, length_dict


def batcher(params, batch):
    sentences = [[params['word2id'][x] for x in ['<start>'] + s + ['<end>']] for s in batch]
    Tensor_dict, length_dict = collate_fn(sentences)

    embeddings = PEmodel.encode(
        params['encoder'],
        Tensor_dict['src'],
        Tensor_dict['mask_src'],
        length_dict
    )
    return embeddings.detach().cpu().numpy()

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


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    init_checkpoint = '/u/lupeng/Project/code/Discourse_summ/saved/octal20'
    with open(os.path.join(init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
        checkpoint = torch.load(os.path.join(init_checkpoint, 'checkpoint'))
    wb = read_pkl(os.path.join(argparse_dict['data_path'], 'vocab_cnt.pkl'))
    word2id = make_vocab(wb, argparse_dict['vocab_size'])
    argparse_dict['word2id'] = len(word2id) + 1
    # weight = np.load(args['weight_path'])
    args = Bunch(argparse_dict)
    model = build_model(args, None)
    model.load_state_dict(checkpoint['model_state_dict'])
    params_senteval['word2id'] = defaultdict(lambda: word2id['<unk>'], word2id)
    params_senteval['encoder'] = model

    logging.info('start evaluating...')
    se = senteval.engine.SE(params_senteval, batcher)
    transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                      'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    for task in transfer_tasks:

        try:
            result = se.eval([task])
            logging.info(f'result of {task}')
            logging.info(result)
        except:
            logging.info(f'{task} failed to be evalatated.')
