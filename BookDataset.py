#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-16

import os, random
from typing import List, Dict
from itertools import chain
from collections import defaultdict, OrderedDict
import json
import re
import pickle

import torch
import torch.utils.data as data

class WikiTextDataset(data.Dataset):
    def __init__(self,
                 split: str,
                 path: str,
                 word2id: Dict[str, int]) -> None:
        assert split in ['train', 'valid', 'test']
        self._data_path = os.path.join(path, split)
        self._n_data = self._count_data(self._data_path)# // 50
        self.word2id = defaultdict(lambda: word2id['<unk>'], word2id)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(os.path.join(self._data_path, f'{i}.json')) as f:
            js = json.loads(f.read())
        src_list = list(map(self.convert2list, js['src']))
        neg_list = list(map(self.convert2list, js['neg']))

        if len(src_list) > 20:
            src_list = src_list[: 20]
        js['src_idx'] = src_list
        js['neg_idx_fwd'] = neg_list[: (len(src_list) - 1)]
        js['neg_idx_bwd'] = neg_list[(len(src_list) - 1): 2 * (len(src_list) - 1)]
        return js

    def convert2list(self, s: str):
        strings = s.lower().split()
        if len(strings) > 50:
            strings = strings[: 50]
        return [self.word2id[w] for w in ['<start>'] + strings + ['<end>']]

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
        def get_idx_by_lens(lens_list: List[int]) -> List[List[int]]:
            idx_list: List[List[int]] = []
            start = 0
            for i in range(len(lens_list)):
                idx_list += [list(range(start, start + lens_list[i]))]
                start = idx_list[-1][-1] + 1
            return idx_list

        def pad_mask(data, name):
            chain_src = list(chain.from_iterable([_[name] for _ in data]))
            src_lens = [len(_) for _ in chain_src]
            max_src_lens = max(src_lens)
            padded_src = torch.zeros(len(chain_src), max_src_lens).long()
            mask_src = torch.zeros(len(chain_src), max_src_lens).long()
            for i, sent in enumerate(chain_src):
                end = src_lens[i]
                padded_src[i, :end] = torch.LongTensor(sent[:end])
                mask_src[i, :end] = 1
            return padded_src, mask_src, src_lens

        src_doc_list: List[int] = []  # count num of sentences in a doc for this batch \
        negf_doc_list: List[int] = []
        negb_doc_list: List[int] = []
        for i, _ in enumerate(data):
            src_doc_list += [len(_['src_idx'])]
            negf_doc_list += [len(_['neg_idx_fwd'])]
            negb_doc_list += [len(_['neg_idx_bwd'])]

        padded_src, mask_src, src_lens = pad_mask(data, 'src_idx')
        padded_nf, mask_nf, nf_lens = pad_mask(data,'neg_idx_fwd')
        padded_nb, mask_nb, nb_lens = pad_mask(data, 'neg_idx_bwd')
        Tensor_dict = {'src': padded_src,
                       # (B x num_src) x max_seq_src_len : num_ is not sure. so (B x num_) is changing
                       'mask_src': mask_src,  # (B x num_src) x max_seq_src_len
                       'nf': padded_nf,
                       'nb': padded_nb,
                       'mnf': mask_nf,
                       'mnb': mask_nb
                       }
        token_dict = {'src': [_['src'] for _ in data]}

        src_idxbylen = get_idx_by_lens(src_doc_list)
        score_idxbylen = get_idx_by_lens([x + 1 for x in src_doc_list])
        # nf_idxbylen = get_idx_by_lens(negf_doc_list)
        # nb_idxbylen = get_idx_by_lens(negb_doc_list)
        # neg_idx = get_neglist(src_doc_list)
        idx_dict = {'rep_idx': src_idxbylen,
                    'score_idx': score_idxbylen,
                    # 'nf_idx': nf_idxbylen,
                    # 'nb_idx': nb_idxbylen
                    }
        # 'neg_idx': neg_idx}
        length_dict = {'src': src_lens,
                       'nf': nf_lens,
                       'nb': nb_lens}
        return Tensor_dict, token_dict, idx_dict, length_dict



def test():
    pass


if __name__ == "__main__":
    test()