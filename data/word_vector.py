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


def get_word2vec(data_path, wordvec_path):
    file_dir = os.path.join(wordvec_path, 'word2vec')
    wb = read_pkl(os.path.join(data_path, 'vocab_cnt.pkl'))
    w2v_file_dir = os.path.join(file_dir, 'word2vec.128d.121k.w2v')

    word2id = make_vocab(wb, 30000)
    with open(w2v_file_dir, 'r') as f:
        lines = f.readlines()
    assert len(word2id) < int(lines[0].split(' ')[0])
    weight = np.zeros((len(word2id), int(lines[0].split(' ')[1])))
    for line in tqdm(lines[1:]):
        key, v = line.split(' ')[0], [float(x) for x in line.split(' ')[1:]]
        if key in word2id:
            weight[word2id[key], :] = v
    path_save = os.path.join(file_dir, 'weight.npy')
    np.save(path_save, weight)
    # with open(path_save, 'wb+') as f:
    #     #pickle.dump(weight, f)
    #     np.save(weight)



def test():
    path = '/u/lupeng/Project/dataset/wikitext-103'
    path2 = '/u/lupeng/Project/code/Discourse_summ'
    get_word2vec(path, path2)
    #with open(, 'r') as f:
    weight = np.load(os.path.join(path2, 'word2vec/weight.npy'))
    print(type(weight))

if __name__ == "__main__":
    test()