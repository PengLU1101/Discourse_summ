#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-10-22

""" pretrain a word2vec on the corpus"""
import argparse
import json, pickle
import logging
import os
from time import time
from collections import Counter
from datetime import timedelta
import re
from tqdm import tqdm

import gensim


try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data

class Sentences(object):
    """ needed for gensim word2vec training"""
    def __init__(self):
        self._path = os.path.join(DATA_DIR, 'train')
        self._n_data = count_data(self._path)

    def __iter__(self):
        for i in range(self._n_data):
            with open(os.path.join(self._path, f'{i}.json') as f:
                data = json.loads(f.read())
            for s in data['src']:
                yield ['<s>'] + s.lower().split() + [r'<\s>']

def get_vocab():
    folder = os.path.join(DATA_DIR, 'train')
    n_data = count_data(folder)
    vocab_counter = Counter()
    print('start building vocab files...')
    for i in tqdm(range(n_data)):
        with open(os.path.join(folder, f'{i}.json')) as f:
            js = json.loads(f.read())
        tokens = ' '.join(js['src']).split()
        tokens = [t.strip() for t in tokens]  # strip
        tokens = [t for t in tokens if t != ""]  # remove empty
        vocab_counter.update(tokens)

    print("Writing vocab file...")
    with open(os.path.join(DATA_DIR, "vocab_cnt.pkl"),
              'wb') as vocab_file:
        pickle.dump(vocab_counter, vocab_file)
    print("Finished writing vocab file")


def main(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    start = time()
    save_dir = args.path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #get_vocab()
    sentences = Sentences()
    model = gensim.models.Word2Vec(
        size=args.dim, min_count=5, workers=16, sg=1)
    model.build_vocab(sentences)
    print(f'vocab built in {timedelta(seconds=time()-start)}')
    model.train(sentences,
                total_examples=model.corpus_count, epochs=model.iter)

    model.save(os.path.join(
        save_dir,
        f'word2vec.{args.dim}d.{len(model.wv.vocab)//1000)}k.bin'
    )
    model.wv.save_word2vec_format(os.path.join(
        save_dir,
        f'word2vec.{args.dim}d.{len(model.wv.vocab)//1000}k.w2v'
    ))

    print(f'word2vec trained in {timedelta(seconds=time()-start)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train word2vec embedding used for model initialization'
    )
    parser.add_argument('--path', default='/u/lupeng/Project/code/Discourse_summ/word2vec', help='root of the model')
    parser.add_argument('--dim', action='store', type=int, default=128)
    args = parser.parse_args()

    main(args)

