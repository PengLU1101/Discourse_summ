#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-09-24
import os
import unittest
import random

import torch

#import read_pkl, Dataset, get_loader




def test():
    try:
        PKL_DIR = os.environ['PKL']
    except KeyError:
        print('please use environment variable to specify .pkl file directories')
    path = os.path.join(PKL, 'data.pkl')
    data = read_pkl(path)
    test_loader = get_loader('cnn_dm', 'test', data['word2id'], data[split+'_token'])
    print(next(test_loader))

if __name__ == "__main__":
    test()