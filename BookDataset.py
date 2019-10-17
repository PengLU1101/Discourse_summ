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

class BookDataset(data.Dataset):
    def __init__(self,
                 split: str,
                 path: str,
                 word2id: Dict[str, int]) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = os.path.join(path, split)
        self._n_data = self._count_data(self._data_path)# // 50
        self.word2id = defaultdict(lambda: word2id['<unk>'], word2id)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(os.path.join(self._data_path, f'{i}.json')) as f:
            js = json.loads(f.read())
        src_list = list(map(self.convert2list, js['article']))
        tgt_list = list(map(self.convert2list, js['summary']))
        neg_list = list(map(self.convert2list, js['neg']))

    def convert2list(self, s: str):
        return [self.word2id[w] for w in ['<start>'] + s.lower().split() + ['<end>']]

    @staticmethod
    def _count_data(path):
        """ count number of data in the given path"""
        matcher = re.compile(r'[0-9]+\.json')
        match = lambda name: bool(matcher.match(name))
        names = os.listdir(path)
        n_data = len(list(filter(match, names)))
        return n_data



def test():
    pass


if __name__ == "__main__":
    test()