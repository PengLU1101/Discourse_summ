#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-07

import json
import os
import pickle
import re
import random
from itertools import chain


import argparse


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
        #slines = fr.readlines()
        #tlines = ft.readlines()
        src_l, tgt_l = [], []
        too_long = []
        idx = 0
        ########################3
        # idx = 171244
        # idx_end = 171553
        # slines = slines[idx: idx_end+1]
        # tlines = tlines[idx: idx_end+1]

        ########################3
        for x, y in tqdm(zip(fr.readlines(), ft.readlines())):
            src_l, tgt_l = [], []
            if len(x) < 1:
                continue
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
            with open(save_json_file, 'r') as f:
                js = json.loads(f.read())
            if len(src_l) > 50:
                src_l = src_l[:50]
            json_file = {'article': src_l, 'summary': _tgt, 'neg': js['neg']}
            with open(save_json_file, 'w+') as f:
                json.dump(json_file, f)
            idx += 1
    # too_long_path = os.path.join(path, split+'.500file.pkl')
    # with open(too_long_path, 'wb+') as f:
    #     pickle.dump(tgt_l, f)

#287228


def add_neg(path, split, part=None, a=None, b=None):
    jsonfile_dir = os.path.join(path, split)
    n_files = len(os.listdir(jsonfile_dir))
    if part:
        start = (part - 1) * (n_files // 30)
        end = min(part * (n_files // 30), n_files)
        print(f"start:{start} \n end:{end}")
    elif a and b:
        print('on a part')
        start = a
        end = b
    else:
        start = 0
        end = n_files
    for i in tqdm(range(start, end)):
        neg_list = []
        with open(os.path.join(jsonfile_dir, f'{i}.json')) as f:
            js = json.loads(f.read())
        # if 'neg' in js:
        #     pass
        # else:
        for idx in range(2 * len(js['article']) - 2):
                neg_idx = random.choice(list(chain(range(0, i), range(i, n_files))))
                with open(os.path.join(jsonfile_dir, f'{neg_idx}.json')) as f:
                    js_neg = json.loads(f.read())

                    neg_sent = random.choice(js_neg['article'])
                neg_list.append(neg_sent)

        js['neg'] = neg_list
        with open(os.path.join(jsonfile_dir, f'{i}.json'), 'w+') as f:
            json.dump(js, f)


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

def check_files(path, split):
    jsonfile_dir = os.path.join(path, split)
    # n_files = len(os.listdir(jsonfile_dir))
    # _ = []
    # for i in tqdm(range(n_files)):
    #     #assert os.path.isfile(os.path.join(jsonfile_dir, f'{i}.json'))
    #     try:
    #         with open(os.path.join(jsonfile_dir, f'{i}.json')) as f:
    #             js = json.loads(f.read())
    #             assert js['article']
    #             assert js['summary']
    #             assert js['neg']
    #     except:
    #         if i not in _:
    #             _.append(i)
    #     if len(js['article']) < 8 and i not in _:
    #         _.append(i)
    #
    #
    # for i in _:
    #     if os.path.isfile(os.path.join(jsonfile_dir, f'{i}.json')):
    #         os.remove(os.path.join(jsonfile_dir, f'{i}.json'))
    ll = os.listdir(jsonfile_dir)
    n_files = len(ll)
    sorted(ll, key=lambda x: x.split(".")[0])
    for i, x in tqdm(enumerate(ll)):
        with open(os.path.join(jsonfile_dir, x)) as f:
            js = json.loads(f.read())

        with open(os.path.join(jsonfile_dir, f'{i}.json'), 'w') as f:
            json.dump(js, f)





if __name__ == "__main__":
    path = "/data/rali5/Tmp/lupeng/data/cnn-dailymail/finished_files"
    path2 = "/data/rali5/Tmp/lupeng/data/new_cnndm"
    #make_json("/data/rali5/Tmp/lupeng/data/new_cnndm", 'val')
    #make_json("/data/rali5/Tmp/lupeng/data/new_cnndm", 'test')
    #make_json("/data/rali5/Tmp/lupeng/data/new_cnndm", 'train')
    #fix_json(path, path2, 'train')
    #show_lens(path2, 'train')
    parser = argparse.ArgumentParser(
        description='data processing',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--a', default=0, type=int)
    parser.add_argument('--b', default=0, type=int)
    args = parser.parse_args()
    add_neg("/data/rali5/Tmp/lupeng/data/new_cnndm", 'train', args.part) #, a=args.a, b=args.b)
    #check_files("/data/rali5/Tmp/lupeng/data/new_cnndm", 'train')
