#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-07

import json
import os
import pickle
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

def fix_json(path, path2, split):
    assert split in ('train', 'val', 'test')
    deliter = re.compile(r"[;!?.]")
    dir_split = os.path.join(path, split)
    dir_list = os.listdir(dir_split)
    for i in tqdm(range(len(dir_list))):
        _new = []
        with open(os.path.join(dir_split, f'{i}.json'), 'r') as f:
            data = json.load(f)
            #summ = data['abstract']
            for sent in data['article']:
                if len(sent.split(" ")) < 100:
                    _new.append(sent)
        new_data = {'article': _new, 'summary': data['abstract']}
        with open(os.path.join(path2, split+f'/{i}.json'), 'w+') as f:
            json.dump(new_data, f)
def show_lens(path, split):
    dir_split = os.path.join(path, split)
    dir_list = os.listdir(dir_split)
    too_long_path = os.path.join(path, split + '.len.pkl')
    ll = []
    for i in tqdm(range(len(dir_list))):
        with open(os.path.join(path, split + f'/{i}.json'), 'r') as f:
            data = json.load(f)
            for sent in data['article']:
                ll.append(len(sent.split(" ")))
    with open(too_long_path, 'wb+') as f:
        pickle.dump(ll, f)

def make_json(path, split):
    assert split in ('train', 'val', 'test')
    deliter = re.compile(r"[;!?]")
    src_path = os.path.join(path, split+'.txt.src')
    tgt_path = os.path.join(path, split+'.txt.tgt.tagged')
    with open(src_path, 'r') as fr, open(tgt_path, 'r') as ft:
        slines = fr.readlines()
        tlines = ft.readlines()
        src_l, tgt_l = [], []
        too_long = []
        idx = 0
        for x, y in tqdm(zip(slines, tlines)):
            if len(x) < 1:
                break
            _src = sent_tokenize(x)
            _tgt = sent_tokenize(y)
            for i, line in enumerate(_src):
                length = len(line.split(" "))
                if length <= 50:
                    src_l.append(line)
                # elif len(line.split(" ")) >= 100:
                #     print(f"the fk line: \n {line}")
                #     splits = re.split(deliter, line)
                #     src_l += norm_line(splits)
                elif length <= 100:
                    splited_line = re.split(deliter, line)
                    for x in splited_line:
                        if len(x.split(' ')) < 50:
                            src_l.append(x + ' .')
                else:
                    pass
            save_json_file = os.path.join(path, split + f'/{idx}.json')
            if len(src_l) > 50:
                src_l = src_l[:50]
            json_file = {'article': src_l, 'summary': _tgt}
            with open(save_json_file, 'w+') as f:
                json.dump(json_file, f)

            idx += 1
    # too_long_path = os.path.join(path, split+'.500file.pkl')
    # with open(too_long_path, 'wb+') as f:
    #     pickle.dump(tgt_l, f)



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
    path = "/data/rali5/Tmp/lupeng/data/cnn-dailymail/finished_files"
    path2 = "/data/rali5/Tmp/lupeng/data/new_cnndm"
    make_json("/data/rali5/Tmp/lupeng/data/new_cnndm", 'train')
    #fix_json(path, path2, 'train')
    #show_lens(path2, 'train')
