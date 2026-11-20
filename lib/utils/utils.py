# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from pathlib import Path

import torch
import torch.optim as optim

from core.config import get_model_name


def create_logger(cfg, cfg_name):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.TRAIN_DATASET
    dataset = dataset.replace(':', '_')
    model, _ = get_model_name(cfg)
    if len(cfg.EXP_SUB_NAME) == 0:
        cfg_name = os.path.basename(cfg_name).split('.')[0]
    else:
        cfg_name = os.path.basename(cfg_name).split('.')[0] + "_" + cfg.EXP_SUB_NAME

    final_output_dir = root_output_dir / dataset / model / cfg_name
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Console-only progress reporting.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    has_console = any(getattr(handler, '_pose_console_handler', False) for handler in logger.handlers)
    if not has_console:
        console = logging.StreamHandler()
        console._pose_console_handler = True
        logger.addHandler(console)

    return logger, str(final_output_dir)


def get_optimizer(cfg, parameters):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            parameters,
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            parameters,
            lr=cfg.TRAIN.LR,
            # betas=(cfg.TRAIN.GAMMA1, cfg.TRAIN.GAMMA2)
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))
