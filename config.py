#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Peng time:2019-07-17


from collections import namedtuple

Config = namedtuple('para',
                    [#  para of embedding layer  #
                     'vocab_size',
                     'emb_dim',

                     #  para of encoder #
                     'encoder_type',
                     'dropout_enc',
                     'nlayers_enc' = 2,
                     'dim_enc' = 128,
                     'heads' = 16,
                     'd_ff' = 8,
                     'dropout_enc' = 0.5,
                     'max_relative_positions' = 100,

                     #  para of parser  #
                     'dim_ipt_prs'=100,
					 'dim_hid_prs'=64,
					 'n_slots'=5,
					 'n_lookback'=1,
					 'resolution'=0.1,
					 'dropout_prs'=0.5,

                     #  para of predictor  #
                     'predictor_type',
                     'dim_pdc',
                     'dim_out'
                     ])
