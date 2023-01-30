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
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--light_heatmap_dir',
                        help='light heatmap dir',
                        required=True,
                        type=str)

    parser.add_argument('--large_heatmap_dir',
                        help='large heatmap dir',
                        required=True,
                        type=str)
    parser.add_argument('--is_light',
                        action='store_true',
                        help='switch to use light heatmap or large heatmap')
    parser.add_argument('--use_both_heatmap',
                        action='store_true',
                        help='switch to use light heatmap and large heatmap')
    parser.add_argument('--light_origin_image_width', type=int, help="width of baseline input rgb image(not heatmap)")
    parser.add_argument('--light_origin_image_height', type=int, help="height of baseline input rgb image(not heatmap)")
    parser.add_argument('--large_origin_image_width', type=int, help="width of baseline input rgb image(not heatmap)")
    parser.add_argument('--large_origin_image_height', type=int, help="height of baseline input rgb image(not heatmap)")

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

    args = parser.parse_args()

    return args


def copy_prev_models(prev_models_dir, model_dir):
    import shutil

    vc_folder = '/hdfs/' \
        + '/' + os.environ['PHILLY_VC']
    source = prev_models_dir
    # If path is set as "sys/jobs/application_1533861538020_2366/models" prefix with the location of vc folder
    source = vc_folder + '/' + source if not source.startswith(vc_folder) \
        else source
    destination = model_dir

    if os.path.exists(source) and os.path.exists(destination):
        for file in os.listdir(source):
            source_file = os.path.join(source, file)
            destination_file = os.path.join(destination, file)
            if not os.path.exists(destination_file):
                print("=> copying {0} to {1}".format(
                    source_file, destination_file))
            shutil.copytree(source_file, destination_file)
    else:
        print('=> {} or {} does not exist'.format(source, destination))


def update_config_pretrained(cfg, pretraind_model):
    cfg.MODEL.PRETRAINED = pretraind_model

def main():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')
    model = eval('models.'+cfg.MODEL.NAME+'.get_transfer_net')(
        cfg, is_train=True
    )

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if cfg.MODEL.USE_WARPING_TRAIN:
      if cfg.MODEL.USE_GT_INPUT_TRAIN:
        dump_input = torch.rand(
          (1, 23, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
        )
      else:
        dump_input = torch.rand(
          (1, 6, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
        )
    else:
      dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
      )

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    ''''
    if args.origin_image_width is not None and args.origin_image_height is not None:
        origin_image_size = (args.origin_image_width, args.origin_image_height)
    else:
        origin_image_size = None
    '''

    light_image_size = (args.light_origin_image_width, args.light_origin_image_height)
    large_image_size = (args.large_origin_image_width, args.large_origin_image_height)


    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True, args.light_heatmap_dir, args.large_heatmap_dir, 
        is_light=args.is_light, light_image_size=light_image_size, 
        large_image_size=large_image_size, use_both_heatmap=args.use_both_heatmap    
    )

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, args.light_heatmap_dir, args.large_heatmap_dir,
        is_light=args.is_light, light_image_size=light_image_size, large_image_size=large_image_size,
        use_both_heatmap=args.use_both_heatmap
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )


    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    optimizer = get_optimizer(cfg, model)

    best_perf = 0.0
    best_model = False
    last_epoch = -1

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    if begin_epoch > 0:
        ckpt_path = os.path.join(final_output_dir, str(begin_epoch) + "_ckpt.pth")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    from core.function_transfer import train
    from core.function_transfer import validate

    
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        epoch_model_stat_file = os.path.join(final_output_dir, str(epoch+1) + 'epoch.pth')
        logger.info('=> saving final model state to {}'.format(
            epoch_model_stat_file)
        )

        torch.save(model.module.state_dict(), epoch_model_stat_file)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }
        save_checkpoint(checkpoint, False, final_output_dir, str(epoch+1) + "_ckpt.pth")

        writer_dict['writer'].close()
    

    epoch_model_stat_file = os.path.join(final_output_dir, '20epoch.pth')

    model = eval('models.'+cfg.MODEL.NAME+'.get_transfer_net')(
        cfg, is_train=False
    )

    model.load_state_dict(torch.load(epoch_model_stat_file), strict=False)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    validate(cfg, valid_loader, valid_dataset, model, criterion,
       final_output_dir, tb_log_dir, writer_dict=writer_dict, validate_type=0)
    
if __name__ == '__main__':
    main()
