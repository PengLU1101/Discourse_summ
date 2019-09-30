#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-08-25
import os
from typing import List, Tuple, Dict, Union, Callable, Iterator
from itertools import chain
from collections import defaultdict, OrderedDict
import json
import re
import pickle

from torch import Tensor as T
import torch
import torch.utils.data as data
from nltk.tokenize import sent_tokenize, word_tokenize



try:
    PKL_DIR = os.environ['PKL_DIR']
except KeyError:
    print('please use environment variable to specify .pkl file directories')



def make_vocab(wc, vocab_size):
    word2id = OrderedDict()
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    word2id['<start>'] = 2
    word2id['<end>'] = 3
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
    return word2id

class CnnDmDataset(data.Dataset):
    def __init__(self,
                 split: str,
                 path: str,
                 word2id: Dict[str, int]) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = os.path.join(path, split)
        self._n_data = self._count_data(self._data_path)
        self.word2id = defaultdict(lambda: word2id['<unk>'], word2id)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(os.path.join(self._data_path, f'{i}.json')) as f:
            js = json.loads(f.read())
        src_list = list(map(self.convert2list, js['article']))
        tgt_list = list(map(self.convert2list, js['abstract']))
        #print(src_list)
        #src_idx = list(map(self.convert2ids, self.word2id, src_list))
        #tgt_idx = list(map(self.convert2ids, self.word2id, tgt_list))
        #print(src_idx)
        js['src_idx'] = src_list
        js['tgt_idx'] = tgt_list
        return js

    def convert2list(self, s: str):
        return [self.word2id[w] for w in ['<start>'] + s.lower().split() + ['<end>']]

    @staticmethod
    def convert2ids(word2id: Dict[str, int],
                    str_list: List[str]):
        return list(map(lambda x: word2id[x], str_list))

    @staticmethod
    def _count_data(path):
        """ count number of data in the given path"""
        matcher = re.compile(r'[0-9]+\.json')
        match = lambda name: bool(matcher.match(name))
        names = os.listdir(path)
        n_data = len(list(filter(match, names)))
        return n_data

    @staticmethod
    def collate_fn(data):
        src_doc_list: List[int] = []  # count num of sentences in a doc for this batch \
        tgt_doc_list: List[int] = []  # both of two list should have same length as batch size.
        for _ in data:
            src_doc_list += [len(_['src_idx'])]
            tgt_doc_list += [len(_['tgt_idx'])]
        chain_src = list(chain.from_iterable([_['src_idx'] for _ in data]))
        chain_tgt = list(chain.from_iterable([_['tgt_idx'] for _ in data]))
        max_src_lens = max(chain.from_iterable(chain_src))
        max_tgt_lens = max(chain.from_iterable(chain_tgt))
        padded_src_lens = [len(_) for _ in chain_src]
        padded_tgt_lens = [len(_) for _ in chain_tgt]

        padded_src = torch.zeros(len(chain_src), max_src_lens).long()
        padded_tgt = torch.zeros(len(chain_tgt), max_tgt_lens).long()
        mask_src = torch.zeros(len(chain_src), max_src_lens).long()
        mask_tgt = torch.zeros(len(chain_tgt), max_tgt_lens).long()

        for i, sent in enumerate(chain_src):
            end = padded_src_lens[i]
            padded_src[i, :end] = torch.LongTensor(sent[:end])
        for i, sent in enumerate(chain_tgt):
            end = padded_tgt_lens[i]
            padded_tgt[i, :end] = torch.LongTensor(sent[:end])
        Tensor_dict = {'src': padded_src,
                       # (B x num_src) x max_seq_src_len : num_ is not sure. so (B x num_) is changing
                       'tgt': padded_tgt,  # (B x num_tgt) x max_seq_tgt_len
                       'mask_src': mask_src,  # (B x num_src) x max_seq_src_len
                       'mask_tgt': mask_tgt  # (B x num_tgt) x max_seq_tgt_len
                       }
        token_dict = {'article': [_['article'] for _ in data],
                      'abstract': [_['abstract'] for _ in data]
                      }
        return Tensor_dict, token_dict


def coll_fn(data):
    source_lists, target_lists = unzip(data)
    # NOTE: independent filtering works because
    #       source and targets are matched properly by the Dataset
    sources = list(filter(bool, concat(source_lists)))
    targets = list(filter(bool, concat(target_lists)))
    assert all(sources) and all(targets)
    return sources, targets


def get_loader(path,
               split,
               batch_size,
               word2id,
               num_workers=10):
    dataset = CnnDmDataset(split, path, word2id)
    if split == 'train':
        shuffle = True
    else:
        shuffle = False
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       collate_fn=dataset.collate_fn)


def test():
    #try:
    #    PKL_DIR = os.environ['PKL_DIR']
    #except KeyError:
    #    print('please use environment variable to specify .pkl file directories')

    def read_pkl(path):
        with open(path, "rb") as f:
            data_dict = pickle.load(f)
        return data_dict
    finished_file_dir = os.path.join(PKL_DIR, 'finished_files')
    wb = read_pkl(os.path.join(finished_file_dir, 'vocab_cnt.pkl'))
    word2id = make_vocab(wb, 30000)

    test_loader = get_loader(finished_file_dir, 'test', 10, word2id)
    #testdata = [iter(_) for _ in test_loader]
    a = next(iter(test_loader))
    print(a)
    b =  next(iter(test_loader))
    print(b)
    assert a == b
if __name__ == "__main__":
    test()