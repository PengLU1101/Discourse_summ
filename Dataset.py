#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-08-25
from typing import List, Tuple, Dict, Union, Callable

from torch import Tensor as T
import torch
import torch.utils.data as data
import itertools import chain


def test():
    pass

def read_pkl(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict

class Dataset(data.Dataset):
    def __init__(self,
                 name: str,
                 split: str,
                 word2id: Dict[str, int],
                 data: Dict[str, List[List[str]]]) -> None:
        self.name = name
        self.split = split
        self.data = data
        assert len(data['src']) == len(data['tgt']), \
            "List of src and tgt must has same length! "
        self.len = len(data['src'])
        self.word2id = word2id


    def __getitem__(self, idx: int) -> Dict[str, Tuple[List[Union[str, int]], List[Union[str, int]]]]:

        src_idx = conver2id('<unk>', self.word2id, self.data['src'][idx])
        tgt_idx = conver2id('<unk>', self.word2id, self.data['tgt'][idx])
        data = {'token': (self.data['src'][idx], self.data['tgt'][idx]),
                'idx': (src_idx, tgt_idx)}
        return data

    def __len__(self) -> int:
        return self.len

    @staticmethod
    def conver2id(unk, word2id, words_list):
        word2id = defaultdict(lambda: unk, word2id)
        return [[word2id[w] for w in words] for words in words_list]
    @staticmethod
    def collate_fn(data: List[Dict[str, Tuple[List[Union[str, int]], List[Union[str, int]]]]]) \
            -> Tuple[Dict[str, T], Dict[str, L]]:
        src_doc_list, tgt_doc_list = [], []
        for _ in data:
            src_doc_list += [len(_['idx'][0])]
            tgt_doc_list += [len(_['idx'][1])]
        chain_src = list(chain.from_iterable([_['idx'][0] for _ in data]))
        chain_tgt = list(chain.from_iterable([_['idx'][1] for _ in data]))
        max_src_lens = max(chain.from_iterable(chain_src))
        max_tgt_lens = max(chain.from_iterable(chain_tgt))
        padded_src_lens = [len(_) for _ in chain_src]
        padded_tgt_lens = [len(_) for _ in chain_tgt]

        padded_src = torch.zeros(len(chain_src, max_src_lens)).long()
        padded_tgt = torch.zeros(len(chain_tgt, max_tgt_lens)).long()
        mask_src = torch.zeros(len(chain_src, max_src_lens)).long()
        mask_tgt = torch.zeros(len(chain_tgt, max_tgt_lens)).long()

        for i, sent in enumerate(chain_src):
            end = padded_src_lens[i]
            padded_src[i, :end] = torch.LongTensor(sent[:end])
        for i, sent in enumerate(chain_tgt):
            end = padded_tgt_lens[i]
            padded_tgt[i, :end] = torch.LongTensor(sent[:end])
        T_dict = {'src': padded_src,
                  'tgt': padded_tgt,
                  'mask_src': mask_src,
                  'mask_tgt': mask_tgt
                  }

        return T_dict, data['token']

def get_loader(name,
               split,
               word2id,
               data):
    dataset = Dataset(name, split, word2id, data)
    if split == 'train':
        shuffle = True
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       collate_fn=dataset.collate_fn)
















if __name__ == "__main__":
    test()