#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-07-16
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import numpy as np
import argparse
import logging
import json, pickle

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import WarmupLinearSchedule, AdamW

import Model
from Dataset import CnnDmDataset, make_vocab
from BookDataset import WikiTextDataset
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
    np.random.seed(1101)
    torch.manual_seed(1101)
    torch.cuda.manual_seed(1101)
    args.do_train = True
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
        logging.info('Gpu is avialable! and set args.device = cuda.')
    args.weight_path = os.path.join(args.data_path, 'word2vec/weight.npy')
    weight = np.load(args.weight_path)

    # Write logs to checkpoint and console
    # Logs details of datasets.
    logging.info(f'Model: {args.model}')
    logging.info(f'Data Path: {args.data_path}')

    wb = read_pkl(os.path.join(args.data_path, 'vocab_cnt.pkl'))
    word2id = make_vocab(wb, args.vocab_size)
    if args.dataset == "wiki":
        args.word2id = len(word2id) + 1
    else:
        args.word2id = len(word2id)
    name2data = {'cnndm': CnnDmDataset, 'book': None, 'wiki': WikiTextDataset} #BookDataset}
    if args.dataset not in name2data:
        raise ValueError('You should use dataset <cnndm>, <wiki> or <book>')

    train_dataset = name2data[args.dataset]('train', args.data_path, word2id)
    val_dataset = name2data[args.dataset]('valid', args.data_path, word2id)
    test_dataset = name2data[args.dataset]('test', args.data_path, word2id)


    logging.info(f'#train: {len(train_dataset)}')
    logging.info(f'#valid: {len(val_dataset)}')
    logging.info(f'#test: {len(test_dataset)}')

    # Logs details of model
    pe_model = Model.build_model(args, weight)

    logging.info('Model Parameter Configuration:')
    #for name, param in pe_model.named_parameters():
    #    logging.info(
    #        f'Parameter {name}: {str(param.size())}, require_grad = {str(param.requires_grad)}'
    #    )

    if torch.cuda.is_available():
        pe_model = pe_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=args.do_train,
                                                   num_workers=max(1, args.cpu_num // 2),
                                                   collate_fn=train_dataset.collate_fn)

        # Set training configuration
        current_learning_rate = args.learning_rate
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, pe_model.parameters()),
                lr=current_learning_rate,
                weight_decay=args.L2,
                momentum=args.momentum
            )
        elif args.optim == 'adam':
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, pe_model.parameters()),
                lr=current_learning_rate,
                weight_decay=args.L2,
            )
        elif args.optim == 'adamw':
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in pe_model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.L2},
                {'params': [p for n, p in pe_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=current_learning_rate, eps=args.adam_epsilon)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_loader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 10
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warm_up_steps, t_total=t_total)
    if args.do_valid:
        valid_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                   batch_size=args.test_batch_size,
                                                   shuffle=False,
                                                   num_workers=max(1, args.cpu_num // 2),
                                                   collate_fn=val_dataset.collate_fn)
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
    if args.do_test:
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.test_batch_size,
                                                  shuffle=False,
                                                  num_workers=max(1, args.cpu_num // 2),
                                                  collate_fn=test_dataset.collate_fn)


    # train_iterator = iter(train_dataloader)
    start_time = time.time()
    step = init_step

    logging.info('Start Training...')
    logging.info(f'init_step = {init_step}')
    logging.info(f'learning_rate = {current_learning_rate}')
    logging.info(f'batch_size = {args.batch_size}')
    logging.info(f'hidden_dim = {args.d_model}')

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        training_logs = []
        writer = SummaryWriter(args.save_path)
        # Training Loop
        train_iter = iter(train_loader)
        for step in range(init_step, args.max_steps):
            #for i in range(len(train_iterator)):
            #pbar = #, total=len(train_loader))
            # train_iterator = iter(train_dataloader)
            start_time = time.time()
            #for step, data in enumerate(train_loader):
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            #for step, data in enumerate(BackgroundGenerator(train_loader)):
            log = pe_model.train_step(pe_model, optimizer, scheduler, data, args, step)
            training_logs.append(log)

            # if step >= warm_up_steps:
            #     current_learning_rate = current_learning_rate / 2
            #     logging.info(f'Change learning_rate to {current_learning_rate} at step {step}')
            #     optimizer = torch.optim.SGD(
            #         filter(lambda p: p.requires_grad, pe_model.parameters()),
            #         lr=current_learning_rate,
            #         weight_decay=args.L2,
            #         momentum=args.momentum
            #     )
            #     warm_up_steps = warm_up_steps * 2

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
                for metric in metrics:
                    writer.add_scalar("train/" + metric, metrics[metric], step)
                current_learning_rate = get_lr(optimizer)
                writer.add_scalar('learning_rate', current_learning_rate, step)

            if args.do_valid and step % args.valid_steps == 0:
                val_logs = []
                logging.info('Evaluating on Valid Dataset...')
                for data in valid_loader:
                    log = pe_model.test_step(pe_model, data, args)
                    val_logs.append(log)
                metrics = {}
                for metric in val_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in val_logs]) / len(val_logs)
                log_metrics('Valid average', step, metrics)
                for metric in metrics:
                    writer.add_scalar("Valid/" + metric, metrics[metric], step)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(pe_model, optimizer, save_variable_list, args)

        if args.do_valid:
            val_logs = []
            logging.info('Evaluating on Valid Dataset...')
            for data in valid_loader:
                log = pe_model.test_step(pe_model, data, args)
                val_logs.append(log)
            metrics = {}
            for metric in val_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in val_logs]) / len(val_logs)
            log_metrics('Valid average', step, metrics)

        if args.do_test:
            test_logs = []
            logging.info('Evaluating on Valid Dataset...')
            for data in test_loader:
                log = pe_model.test_step(pe_model, data, args)
                test_logs.append(log)
            metrics = {}
            for metric in test_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in test_logs]) / len(test_logs)
            log_metrics('test average', step, metrics)
            for metric in metrics:
                writer.add_scalar("test/"+metric, metrics[metric], step)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = pe_model.test_step(pe_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == "__main__":
    main(parse_args())