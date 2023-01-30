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
from networks import CoordRegressionNetwork
import torch.nn as nnz
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',

                        help='experiment configure file name',
                        required=True,
                        type=str)

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

    parser.add_argument('--skip_train',
                        action='store_true')

    parser.add_argument('--skip_validate',
                        action='store_true')




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


def main():
    args = parse_args()

    update_config(cfg, args)

    if args.prevModelDir and args.modelDir:
        # copy pre models for philly
        copy_prev_models(args.prevModelDir, args.modelDir)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    #logger.info(pprint.pformat(args))
    #logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    modeltype = "mobilenetv2"
    model = CoordRegressionNetwork(n_locations=17, backbone=modeltype) # .to(device)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    pretrained_dict = torch.load(cfg.MODEL.PRETRAINED)
    # pretrained_dict = pre_net.state_dict()
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != "hm_conv.weight"}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.module.load_state_dict(pretrained_dict, strict=False)

    # model.module.load_state_dict(pre_net, strict=False)
    model.train()

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # criterion = nn.MSELoss().cuda()
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR)


    ### importing train/validate functions
    if not cfg.MODEL.SPATIOTEMPORAL_POSE_AGGREGATION:
       from core.function import train
       from core.function import validate
    else:
       from core.function_PoseAgg import train
       from core.function_PoseAgg import validate
    ####

    if not args.skip_train:
        for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
            lr_scheduler.step()

            # train for one epoch
            train(cfg, train_loader, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict, mobilenet=True)

            epoch_model_state_file = os.path.join(
                final_output_dir, str(epoch+1) + 'epoch.pth'
            )
            logger.info('=> saving epoch model state to {}'.format(
                epoch_model_state_file)
            )
            torch.save(model.module.state_dict(), epoch_model_state_file)
    

    epoch_model_stat_file = os.path.join(final_output_dir, '100epoch.pth')

    model = CoordRegressionNetwork(n_locations=17, backbone=modeltype) # .to(device)

    model.load_state_dict(torch.load(epoch_model_stat_file), strict=False)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    if not args.skip_validate:
        validate(cfg, valid_loader, valid_dataset, model, criterion,
        final_output_dir, tb_log_dir, writer_dict=writer_dict, mobilenet=True)


    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )

    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
