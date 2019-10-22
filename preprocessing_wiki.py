#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-20

import json
import os
import pickle
import re
import random
from itertools import chain

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

def make_json(path_data, split):
    input_file = os.path.join(path_data, f'wiki.{split}.tokens')
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    save_id = 0
    print(f'start preprossing files with {len(lines)} lines...')
    for line in tqdm(lines):
        if len(line) > 500:
            sentences = sent_tokenize(line)
            json_file = os.path.join(path_data, f'{split}/{save_id}.json')
            if (len(sentences) > 5) and (len(sentences) < 50):
                new_data = {'src': sentences}
                with open(json_file, 'w+') as fw:
                    json.dump(new_data, fw)
                save_id += 1
            elif len(sentences) > 50:
                new_data = {'src': sentences[: 50]}
                with open(json_file, 'w+') as fw:
                    json.dump(new_data, fw)
                save_id += 1

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

        for idx in range(2 * len(js['src']) - 2):
                neg_idx = random.choice(list(chain(range(0, i), range(i, n_files))))
                with open(os.path.join(jsonfile_dir, f'{neg_idx}.json')) as f:
                    js_neg = json.loads(f.read())

                    neg_sent = random.choice(js_neg['src'])
                neg_list.append(neg_sent)

        js['neg'] = neg_list
        with open(os.path.join(jsonfile_dir, f'{i}.json'), 'w+') as f:
            json.dump(js, f)

def test():
    pass


if __name__ == "__main__":
    path = '/u/lupeng/Project/dataset/wikitext-103'
    #make_json(path, 'train')
    add(path, 'train')
