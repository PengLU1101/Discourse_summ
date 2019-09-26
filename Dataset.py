#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-08-25
import os
from typing import List, Tuple, Dict, Union, Callable, Iterator
from itertools import chain
from collections import defaultdict

from torch import Tensor as T
import torch
import torch.utils.data as data


try:
    PKL_DIR = os.environ['PKL']
except KeyError:
    print('please use environment variable to specify .pkl file directories')
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
                 data: Dict[str, List[List[List[str]]]]) -> None:
        self.name = name
        self.split = split
        self.data = data
        assert len(data['src']) == len(data['tgt']), \
            "List of src and tgt must has same length! "
        self.len = len(data['src'])
        self.word2id = word2id


    def __getitem__(self, idx: int) \
            -> Dict[str, List[List[Union[str, int]]]]:
        src_idx = self.conver2id('<unk>', self.word2id, self.data['src'][idx])
        tgt_idx = self.conver2id('<unk>', self.word2id, self.data['tgt'][idx])
        data = {'src': self.data['src'][idx],
                'tgt': self.data['tgt'][idx],
                'src_idx': src_idx,
                'tgt_idx': tgt_idx}
        return data

    def __len__(self) -> int:
        return self.len

    @staticmethod
    def conver2id(unk: str,
                  word2id: Dict[str, int],
                  words_list: List[List[str]]) -> List[List[int]]:
        word2id = defaultdict(lambda: word2id[unk], word2id)
        return [[word2id[w] for w in words] for words in words_list]

    @staticmethod
    def collate_fn(data: List[Dict[str, Union[List[List[str]], List[List[int]]]]]) \
            -> Tuple[Dict[str, T], Dict[str, List[List[List[int]]]]]:
        src_doc_list: List[int] = [] # count num of sentences in a doc for this batch \
        tgt_doc_list: List[int] = [] # both of two list should have same length as batch size.
        for _ in data:
            src_doc_list += [len(_['src_idx'])]
            tgt_doc_list += [len(_['tgt_idx'])]
        chain_src = list(chain.from_iterable([_['src_idx'] for _ in data]))
        chain_tgt = list(chain.from_iterable([_['tgt_idx'] for _ in data]))
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
        Tensor_dict = {'src': padded_src, # (B x num_src) x max_seq_src_len : num_ is not sure. so (B x num_) is changing
                       'tgt': padded_tgt, # (B x num_tgt) x max_seq_tgt_len
                       'mask_src': mask_src, # (B x num_src) x max_seq_src_len
                       'mask_tgt': mask_tgt # (B x num_tgt) x max_seq_tgt_len
        }
        token_dict = {'src': [_['src'] for _ in data],
                      'tgt': [_['tgt'] for _ in data]
        }
        return Tensor_dict, token_dict

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