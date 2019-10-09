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
    deliter = re.compile(r"([.!！?？])")
    src_path = os.path.join(path, split+'.txt.src')
    tgt_path = os.path.join(path, split+'.txt.tgt.tagged')
    with open(src_path, 'r') as fr, open(tgt_path, 'r') as ft:
        slines = fr.readlines()
        tlines = ft.readlines()
        src_l, tgt_l = [], []
        too_long = []
        for x, y in tqdm(zip(slines, tlines)):
            _src = sent_tokenize(x)
            _tgt = sent_tokenize(y)
            for i, line in enumerate(_src):
                if len(line.split(" ")) <= 50:
                    src_l.append(line)
                # elif len(line.split(" ")) >= 100:
                #     print(f"the fk line: \n {line}")
                #     splits = re.split(deliter, line)
                #     src_l += norm_line(splits)
                else:
                    too_long.append(len(line.split(" ")))
    too_long_path = os.path.join(path, split+'.toolong.pkl')
    with open(too_long_path, 'w+') as f:
        pickle.dump(too_long, too_long_path)


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
        if len(x.split(' ')) > 50:
            _[i] = ' '.join(x.split(' ')[:50])
    for i, x in enumerate(_):
        print(f'{i} is: \n {x}')
    return _



if __name__ == "__main__":
    make_json("/data/rali5/Tmp/lupeng/data/new_cnndm", 'val')