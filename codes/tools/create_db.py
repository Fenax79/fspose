# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
#from core.function import train
#from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models
import copy
from networks import CoordRegressionNetwork
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--mobilenet', action="store_true")
    parser.add_argument('--save_accuracy_folder', type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--heatmap_root',
                        help='heatmap output dir',
                        type=str,
                        default='')                    
    parser.add_argument('--do_validate',
                        action='store_true')
    parser.add_argument('--skip_create',
                        action='store_true')


    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    using_dataset = "posetrack_same_person"

    if args.prevModelDir and args.modelDir:
        # copy pre models for philly
        copy_prev_models(args.prevModelDir, args.modelDir)


    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if args.mobilenet:
        modeltype = "mobilenetv2"
        model = CoordRegressionNetwork(n_locations=17, backbone=modeltype) # .to(device)
    else:
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=False
        )

    model_state_file = cfg.MODEL.PRETRAINED
    model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = eval('dataset.'+using_dataset)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )


    train_dataset = eval('dataset.'+using_dataset)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    if not args.skip_create:
        logger, final_output_dir, tb_log_dir = create_logger(
            cfg, args.cfg,"train")

        logger.info('########################################')

        
        from core.function import create_heatmap_and_save
        create_heatmap_and_save(cfg, train_loader, train_dataset, model, criterion,
                final_output_dir, tb_log_dir, args.heatmap_root, mobilenet=args.mobilenet)
        
        logger, final_output_dir, tb_log_dir = create_logger(
            cfg, args.cfg, 'valid')
        logger.info('########################################')

        create_heatmap_and_save(cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, args.heatmap_root, mobilenet=args.mobilenet)
    
    
    if args.do_validate:
        logger, final_output_dir, tb_log_dir = create_logger(
            cfg, args.cfg, 'valid')
        logger.info('########################################')

        from core.function import validate
        validate(cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, mobilenet=args.mobilenet, #input_heatmap_dir=args.heatmap_root, 
                save_accuracy_folder=args.save_accuracy_folder)
    

if __name__ == '__main__':
    main()
