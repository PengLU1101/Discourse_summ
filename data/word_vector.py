#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-08-09

import os
import numpy as np
import pickle
from collections import defaultdict, OrderedDict
from tqdm import tqdm

def read_pkl(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


def make_vocab(wc, vocab_size):
    word2id = OrderedDict()
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    word2id['<start>'] = 2
    word2id['<end>'] = 3
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
    return word2id


def get_word2vec(path):
    finished_file_dir = os.path.join(path, 'finished_files')
    wb = read_pkl(os.path.join(finished_file_dir, 'vocab_cnt.pkl'))
    w2v_file_dir = os.path.join(path, 'word2vec/word2vec.128d.226k.w2v')

    word2id = make_vocab(wb, 30000)
    with open(w2v_file_dir, 'r') as f:
        lines = f.readlines()
    assert len(word2id) < int(lines[0].split(' ')[0])
    weight = np.zeros((len(word2id), int(lines[0].split(' ')[1])))
    for line in tqdm(lines[1:]):
        key, v = line.split(' ')[0], [float(x) for x in line.split(' ')[1:]]
        if key in word2id:
            weight[word2id[key], :] = v

    path_save = os.path.join(path, 'word2vec/weight.pkl')
    with open(path_save, 'wb+') as f:
        pickle.dump(weight, f)



def test():
    path = '/data/rali5/Tmp/lupeng/data/cnn-dailymail'
    get_word2vec(path)


if __name__ == "__main__":
    test()