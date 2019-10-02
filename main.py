#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-07-16

import os
import random
import numpy as np
import argparse
import logging
import json

import torch
import numpy as np

import Model
from Dataset import CnnDmDataset
from Parser import *

# try:
#     DATA_DIR = os.environ['PKL_DIR']
# except KeyError:
#     print('please use environment variable to specify .pkl file directories')

def main(args):
    # train/test
    # data sir/model dir/ checkpoint dir
    # prepare dataset
    # build model/optimizer
    #
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    set_logger(args)
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        logging.info('Gpu is avialable! and set args.device = cuda.')

    # Write logs to checkpoint and console
    # Logs details of datasets.
    logging.info(f'Model: {args.model}')
    logging.info(f'Data Path: {args.data_path}')

    finished_file_dir = os.path.join(args.data_path, 'finished_files')
    wb = read_pkl(os.path.join(finished_file_dir, 'vocab_cnt.pkl'))
    word2id = make_vocab(wb, args.vocab_size)
    args.word2id = len(word2id)
    name2data = {'cnndm': CnnDmDataset, 'book': BookDataset}
    if args.dataset not in name2data:
        raise ValueError('You should use dataset <cnndm> or <book>')

    train_dataset = name2data[args.dataset]('train', finished_file_dir, word2id)

    logging.info(f'#train: {len(train_dataset)}')
    logging.info(f'#valid: {len(val_dataset)}')
    logging.info(f'#test: {len(test_dataset)}')

    # Logs details of model
    pe_model = Model.build_model(args)

    logging.info('Model Parameter Configuration:')
    for name, param in pe_model.named_parameters():
        logging.info(
            f'Parameter {name}: {str(param.size())}, require_grad = {str(param.requires_grad)}'
        )

    if torch.cuda.is_available():
        pe_model = pe_model.to(args.device )

    if args.do_train:
        # Set training dataloader iterator
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=args.do_train,
                                                   num_workers=max(1, args.cpu_num // 2),
                                                   collate_fn=dataset.collate_fn)
        train_iterator = iter(train_dataloader)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, pe_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        pe_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing {args.model} Model...')
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info(f'init_step = {init_step}')
    logging.info(f'learning_rate = {current_learning_rate}')
    logging.info(f'batch_size = {args.batch_size}')
    logging.info(f'hidden_dim = {args.d_model}')

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        training_logs = []

        # Training Loop
        for step in range(init_step, args.max_steps):
            #for i in range(len(train_iterator)):
            log = pe_model.train_step(pe_model, optimizer, train_iterator, args)


            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info(f'Change learning_rate to {current_learning_rate} at step {step}')
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, pe_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(pe_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = pe_model.test_step(pe_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(pe_model, optimizer, save_variable_list, args)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = pe_model.test_step(pe_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = pe_model.test_step(pe_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = pe_model.test_step(pe_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == "__main__":
    main(parse_args())