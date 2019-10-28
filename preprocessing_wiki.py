#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-20

import json
import os
import random
from itertools import chain
import argparse
import re

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
            if (len(sentences) > 8) and (len(sentences) < 50):
                new_data = {'src': sentences}
                with open(json_file, 'w+') as fw:
                    json.dump(new_data, fw)
                save_id += 1
            elif len(sentences) > 50:
                new_data = {'src': sentences[: 50]}
                with open(json_file, 'w+') as fw:
                    json.dump(new_data, fw)
                save_id += 1
    print('end')

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
    print(f'start add neg example for {split} files ...')
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
def finegrain(path1, path2, split):
    read_path = os.path.join(path1, split)
    save_path = os.path.join(path2, split)
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(read_path)
    n_data = len(list(filter(match, names)))
    idx = 0
    for i in tqdm(range(n_data)):
        with open(os.path.join(read_path, f"{i}.json")) as f:
            js = json.loads(f.read())
        _ = [len(x.lower().split()) for x in js['src']]
        if max(_) < 50:
            with open(os.path.join(save_path, f'{idx}.json'), "w+") as f:
                json.dump(js, f)
            idx += 1




def check_files(path, split):
    jsonfile_dir = os.path.join(path, split)
    n_files = len(os.listdir(jsonfile_dir))
    _ = []
    for i in tqdm(range(n_files)):
        #assert os.path.isfile(os.path.join(jsonfile_dir, f'{i}.json'))
        try:
            with open(os.path.join(jsonfile_dir, f'{i}.json')) as f:
                js = json.loads(f.read())
                assert js['src']
                assert js['neg']
        except:
            if i not in _:
                _.append(i)

    #print(_)
    for i in tqdm(_):
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
    #
    # for i in _:
    #     if os.path.isfile(os.path.join(jsonfile_dir, f'{i}.json')):
    #         os.remove(os.path.join(jsonfile_dir, f'{i}.json'))
    # ll = os.listdir(jsonfile_dir)
    # n_files = len(ll)
    # sorted(ll, key=lambda x: x.split(".")[0])
    # for i, x in tqdm(enumerate(ll)):
    #     with open(os.path.join(jsonfile_dir, x)) as f:
    #         js = json.loads(f.read())
    #
    #     with open(os.path.join(jsonfile_dir, f'{i}.json'), 'w') as f:
    #         json.dump(js, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='data processing',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--a', default=0, type=int)
    parser.add_argument('--b', default=0, type=int)
    args = parser.parse_args()
    path = '/u/lupeng/Project/dataset/wikitext-103'
    path2 = '/u/lupeng/Project/dataset/wikitext_103'
    #make_json(path, 'train')
    #add_neg(path, 'train', part=args.part)
    #check_files(path, 'train')
    finegrain(path, path2, 'train')