#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-08-25
import os
from typing import List, Tuple, Dict, Union, Callable, Iterator
from itertools import chain
from collections import defaultdict
import json
import re

from torch import Tensor as T
import torch
import torch.utils.data as data
from nltk.tokenize import sent_tokenize, word_tokenize



try:
    PKL_DIR = os.environ['PKL_DIR']
except KeyError:
    print('please use environment variable to specify .pkl file directories')


def read_pkl(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict

def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
        id2word[i] =
    return word2id

class CnnDmDataset(Dataset):
    def __init__(self,
                 split: str,
                 path: str,
                 word2id: Dict[str, int]) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = os.path.join(path, split)
        self._n_data = self._count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(os.path.join(self._data_path, f'{i}.json')) as f:
            js = json.loads(f.read())
        src_list = map(self.convert2list, js['article'])
        tgt_list = map(self.convert2list, js['abstract'])
        src_idx = list(map(self.convert2ids, src_list))
        tgt_idx = list(map(self.convert2ids, tgt_list))
        js{'src_idx': src_idx,
           'tgt_idx': tgt_idx}
        return js

    @staticmethod
    def convert2list(s: str):
        return ['<s>'] + s.lower().split() + [r'<\s>']

    @staticmethod
    def convert2ids(word2id: Dict[str, int],
                    str_list: List[str]):
        return map(lambda x: word2id[x], str_list)

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

def


def get_loader(name,
               split,
               batch_size,
               shuffle,
               word2id,
               data):
    dataset = Dataset(name, split, word2id, data)
    if split == 'train':
        shuffle = True
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       #num_workers=1,
                                       collate_fn=dataset.collate_fn)

class Dataset(data.Dataset):
    def __init__(self,
                 name: str,
                 split: str,
                 word2id: Dict[str, int],
                 data: Dict[str, str]) -> None:
        self.name = name
        self.split = split
        self.data = data
        with open(os.path.join(path, data['src']), 'r') as f:
            self.src = f.readlines()
        with open(os.path.join(path, data['tgt']), 'r') as f:
            self.tgt = f.readlines()
        self.len = len(self.src)
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
    def get_tokenized_sent(sent_lists):
        if len(sent_lists):
            return [["<sent>"] + word_tokenize(x) + ["</sent>"] for x in sent_lists]
        else:
            return []
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
        Tensor_dict = {'src': padded_src, # (B x num_src) x max_seq_src_len : num_ is not sure. so (B x num_) is changing
                       'tgt': padded_tgt, # (B x num_tgt) x max_seq_tgt_len
                       'mask_src': mask_src, # (B x num_src) x max_seq_src_len
                       'mask_tgt': mask_tgt # (B x num_tgt) x max_seq_tgt_len
        }
        token_dict = {'src': [_['src'] for _ in data],
                      'tgt': [_['tgt'] for _ in data]
        }
        return Tensor_dict, token_dict





def test():
    #try:
    #    PKL_DIR = os.environ['PKL_DIR']
    #except KeyError:
    #    print('please use environment variable to specify .pkl file directories')
    path = os.path.join(PKL_DIR, 'data.pkl')
    data = read_pkl(path)
    test_loader = get_loader('cnn_dm', 'test', 32, True, data['word2id'], data['test'+'_token'])
    testdata = [iter(_) for _ in test_loader]
    print(len(testdata))

if __name__ == "__main__":
    test()