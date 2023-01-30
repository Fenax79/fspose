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

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
#from core.function import validate
from utils.utils import create_logger

import dataset
import models
import logging
import numpy as np
import cv2

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

    parser.add_argument('--input',
                        help='input image size must be input size of model',
                        type=str,
                        default='')

    parser.add_argument('--input_sup',
                        help='input image size must be input size of model',
                        type=str,
                        default=None)

    # ignore
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

    args = parser.parse_args()
    return args


def read_image(image_path, cfg):
    r = open(image_path,'rb').read()
    img_array = np.asarray(bytearray(r), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if cfg.DATASET.COLOR_RGB:
        data_numpy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return data_numpy




def main():
    args = parse_args()
    update_config(cfg, args)

    logger = logging.getLogger(__name__)
    logger.info('########################################')

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    input_ = read_image(args.input, cfg)
    # x = 40 # 30 # 190
    # y = 170 # 0 # 120
    # input_ = input_[y:y+256, x:x+192, :]
    input_base = np.zeros((256, 192, 3), dtype=np.uint8)
    h,w = input_.shape[:2]
    width = 192
    height = round(h * (width / w))
    dst = cv2.resize(input_, dsize=(width, height))
    y_start = int((256 - height) / 2)
    input_base[y_start:y_start+height, :, :] = dst[:, :, :]
    input_  = input_base

    cv2.imshow("input_", cv2.cvtColor(input_, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    input_ = transform(input_)
    input_ = torch.reshape(input_, (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))

    if args.input_sup is not None:
        input_sup = read_image(args.input_sup, cfg)
        input_sup = transform(input_sup)
        input_sup = torch.reshape(input_sup, (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
        concat_input = torch.cat((input, input_sup), 1)

    outputs = model(input_)
    logger.info(outputs.shape)

    for i in range(1):
        num_0_heatmap = outputs[i][0].to('cpu').detach().numpy().copy()
        total = np.zeros(num_0_heatmap.shape, dtype=num_0_heatmap.dtype)

        for j in range(cfg.MODEL.NUM_JOINTS):
            '''
            filepath = os.path.join(parent, str(j)+ ".png")
            img = outputs[i][j].to('cpu').detach().numpy().copy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            '''
            total += outputs[i][j].to('cpu').detach().numpy().copy()
            # cv2.imshow("img", img)
            # cv2.waitKey(0)

            # cv2.imwrite(filepath, img)
        
        # filepath = os.path.join(parent, "total.png")
        total = np.clip(total * 255, 0, 255).astype(np.uint8)
        # cv2.imwrite(filepath, total)
        cv2.imshow("total", total)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
