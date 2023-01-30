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
from config import update_config, update_config_light
from core.loss import JointsMSELoss
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models
from networks import CoordRegressionNetwork
import torch.nn as nnz

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--cfg_lightx2',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('--lightx2_model', type=str)

    parser.add_argument('--light_heatmap_dir', help='light heatmap dir', required=True, type=str)
    parser.add_argument('--large_heatmap_dir', help='large heatmap dir', required=True, type=str)
    parser.add_argument('--inh_heatmap_dir', help='inh heatmap dir', type=str)
    parser.add_argument('--light_origin_image_width', required=True, type=int)
    parser.add_argument('--light_origin_image_height', required=True, type=int)
    parser.add_argument('--large_origin_image_width', required=True, type=int)
    parser.add_argument('--large_origin_image_height', required=True, type=int)
    parser.add_argument('--three_hp', action="store_true")
    parser.add_argument('--inh_hp', action="store_true")
    parser.add_argument('--write_gt', action="store_true")
    parser.add_argument('--save_accuracy_folder', type=str)
    parser.add_argument('--detail_info_file', type=str)
    parser.add_argument('--i_frame_intreval', type=int, default=16)
    parser.add_argument('--validate_type', type=int, default=0)
    parser.add_argument('--save_all_score', action="store_true")
    parser.add_argument('--half', action="store_true")
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
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

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_light = cfg.clone()
    update_config(cfg, args)

    if args.cfg_lightx2 is not None:
        update_config_light(cfg_light, args.cfg_lightx2, pretrain_model=args.lightx2_model)

    light_image_size = (args.light_origin_image_width, args.light_origin_image_height)
    large_image_size = (args.large_origin_image_width, args.large_origin_image_height)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'validate')

    model = eval('models.'+cfg.MODEL.NAME+'.get_transfer_net')(
        cfg, is_train=False
    )

    if args.cfg_lightx2 is not None:
        model_light = eval('models.'+cfg_light.MODEL.NAME+'.get_transfer_net')(
            cfg_light, is_train=False
        )

    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, args.light_heatmap_dir, args.large_heatmap_dir,
        inh_heatmap_dir=args.inh_heatmap_dir, three_hp=args.three_hp, inh_hp=args.inh_hp, 
        light_image_size=light_image_size, large_image_size=large_image_size, 
        i_frame_intreval=args.i_frame_intreval
    )

    model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED), strict=False)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    if args.cfg_lightx2 is not None:
        model_light.load_state_dict(torch.load(cfg_light.MODEL.PRETRAINED), strict=False)
        model_light = torch.nn.DataParallel(model_light, device_ids=cfg_light.GPUS).cuda()

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )


    from core.function_transfer import validate_inh_hp
    from core.function_transfer import validate_3heatmap
    valtype3_light_size = (int(light_image_size[0] / 4), int(light_image_size[1] / 4))
    valtype3_large_size = (int(large_image_size[0] / 4), int(large_image_size[1] / 4))

    assert not args.inh_hp
    if args.inh_hp:
        validate_inh_hp(cfg, valid_loader, valid_dataset, model, criterion, 
           final_output_dir, tb_log_dir, validate_type=args.validate_type, 
           save_accuracy_folder=args.save_accuracy_folder, write_gt=args.write_gt,
            detail_info_file=args.detail_info_file, i_frame_intreval=args.i_frame_intreval, save_all_score=args.save_all_score,
            model_light=model_light, light_hp_size=valtype3_light_size, large_hp_size=valtype3_large_size)

    if args.three_hp:
        validate_3heatmap(cfg, valid_loader, valid_dataset, model, criterion, 
            final_output_dir, tb_log_dir, validate_type=args.validate_type, 
            save_accuracy_folder=args.save_accuracy_folder, write_gt=args.write_gt, 
            detail_info_file=args.detail_info_file, i_frame_intreval=args.i_frame_intreval, save_all_score=args.save_all_score,
            model_light=model_light,light_hp_size=valtype3_light_size, large_hp_size=valtype3_large_size, half=args.half)
    

if __name__ == '__main__':
    main()
