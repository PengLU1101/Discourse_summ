#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-07

import json
import os
import pickle
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

def make_json(path, split):
    assert split in ('train', 'val', 'test')
    deliter = re.compile(r"([.!！?？；;])")
    src_path = os.path.join(path, split+'.txt.src')
    tgt_path = os.path.join(path, split+'.txt.tgt.tagged')
    with open(src_path, 'r') as fr, open(tgt_path, 'r') as ft:
        slines = fr.readlines()
        tlines = ft.readlines()
        src_l, tgt_l = [], []
        for x, y in tqdm(zip(slines, tlines)):
            _src = sent_tokenize(x)
            _tgt = sent_tokenize(y)
            for i, line in enumerate(_src):
                if len(line.split(" ")) <= 40:
                    src_l.append(line)
                else:
                    splits = re.split(deliter, line)
                    src_l + = norm_line(splits)

def norm_line(line):
    if len(line) % 2 != 0:
        line = line[:-1]
    o = line[1::2]
    e = line[0::2]
    if max(e) > max(o):
        _ = [x + ' ' + y for x, y in zip(e, o)]
    else:
        _ = [x + ' ' + y for x, y in zip(o, e)]
    for i, x in enumerate(_):
        if len(x.split(' ')) > 40:
            _[i] = ' '.join(x.split(' '))
    return






with open()
if __name__ == "__main__":
    test()