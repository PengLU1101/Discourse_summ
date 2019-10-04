#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-09-30

import argparse
import json, pickle
import os
import logging
import torch


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Discourse Sentence Representations Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--model', type=str, default='PE_Model')

    parser.add_argument('--data_path', type=str, default='/data/rali5/Tmp/lupeng/data/cnn-dailymail')
    parser.add_argument('--dataset', type=str, default='cnndm', help='cnndm or book')
    parser.add_argument('-save', '--save_path', default='/u/lupeng/Project/code/Discourse_summ/saved', type=str)
    #parser.add_argument()

    parser.add_argument('--score_type', default='dot', type=str)
    parser.add_argument('-v', '--vocab_size', default=30000, type=int)
    parser.add_argument('-ed', '--emb_dim', default=256, type=int)
    parser.add_argument('-md', '--d_model', default=256, type=int)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-t', '--resolution', default=.5, type=float)
    parser.add_argument('--hard', default=True, type=str)
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--n_layer', default=2, type=int)
    #parser.add_argument('--weight_path', default='')

    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')

    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    return parser.parse_args()


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    print(argparse_dict)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def read_pkl(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict