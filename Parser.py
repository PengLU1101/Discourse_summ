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
    parser.add_argument('--machine', type=str, default='octal19')

    #parser.add_argument('--data_path', type=str, default='/data/rali5/Tmp/lupeng/data/new_cnndm')
    parser.add_argument('--data_path', type=str, default='/u/lupeng/Project/dataset/wikitext-103')
    parser.add_argument('--dataset', type=str, default='wiki', help='cnndm or book')
    parser.add_argument('-save', '--save_path', default='/u/lupeng/Project/code/Discourse_summ/saved', type=str)
    #parser.add_argument()
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=10, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--score_type_parser', default='dot', type=str)
    parser.add_argument('--score_type_predictor', default='denselinear', type=str)
    parser.add_argument('--encoder_type', default='transformer', type=str)
    parser.add_argument('-v', '--vocab_size', default=30000, type=int)
    parser.add_argument('-ed', '--emb_dim', default=128, type=int)
    parser.add_argument('-md', '--d_model', default=512, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-t', '--resolution', default=0.1, type=float)
    parser.add_argument('--hard', default=True, type=str)
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--L2', default=0.0, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--n_layer', default=1, type=int)
    parser.add_argument('--bidirectional', default=True, type=bool)
    parser.add_argument('--bidirectional_compute', default=True, type=bool)
    #parser.add_argument('--weight_path', default='')

    parser.add_argument('-r', '--regularization', default=1.0, type=float)
    parser.add_argument('--test_batch_size', default=10, type=int, help='valid/test batch size')

    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--max_steps', default=3000000, type=int)
    parser.add_argument('--warm_up_steps', default=3000, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=10, type=int, help='train log every xx steps')
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
    save_path = os.path.join(args.save_path, args.machine)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    with open(os.path.join(save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_path, 'checkpoint')
    )


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    if not args.init_checkpoint:
        save_path = os.path.join(args.save_path, args.machine)
    else:
        save_path = os.path.join(args.init_checkpoint, args.machine)

    if args.do_train:
        log_file = os.path.join(save_path, 'train.log')
    else:
        log_file = os.path.join(save_path, 'test.log')

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
