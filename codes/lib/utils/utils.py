# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import numpy as np
import cv2

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        #print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True)

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

#    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    #print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
#        if cfg.MODEL.FREEZE_WEIGHTS:
        if False:
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.TRAIN.LR
            )
#            optimizer = optim.SGD(
#              filter(lambda p: p.requires_grad, model.parameters()),
#              lr=cfg.TRAIN.LR,
#              momentum=cfg.TRAIN.MOMENTUM,
#              weight_decay=cfg.TRAIN.WD,
#              nesterov=cfg.TRAIN.NESTEROV
#            )

#        elif cfg.MODEL.USE_STSN_TRAIN:
#            #print(xy)
#            optimizer = optim.Adam([
#                {'params': model.module.conv1.parameters()},
#                {'params': model.module.bn1.parameters()},
#                {'params': model.module.conv2.parameters()},
#                {'params': model.module.bn2.parameters()},
#                {'params': model.module.layer1.parameters()},
#                {'params': model.module.transition1.parameters()},
#                {'params': model.module.stage2.parameters()},
#                {'params': model.module.transition2.parameters()},
#                {'params': model.module.stage3.parameters()},
#                {'params': model.module.transition3.parameters()},
#                {'params': model.module.offsets1.parameters(), 'lr': cfg.TRAIN.STSN_LR},
#                {'params': model.module.offsets2.parameters(), 'lr': cfg.TRAIN.STSN_LR},
#                {'params': model.module.offsets3.parameters(), 'lr': cfg.TRAIN.STSN_LR},
#                {'params': model.module.offsets4.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.embed1.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.embed2.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.embed3.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.embed4.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.cos_sim1.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.cos_sim2.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.cos_sim3.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.cos_sim4.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.channel_softmax1.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.channel_softmax2.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.channel_softmax3.parameters(), 'lr': cfg.TRAIN.STSN_LR},
##                {'params': model.module.channel_softmax4.parameters(), 'lr': cfg.TRAIN.STSN_LR},
#                {'params': model.module.deform_conv1.parameters()},
#                {'params': model.module.deform_conv2.parameters()},
#                {'params': model.module.deform_conv3.parameters()},
#                {'params': model.module.deform_conv4.parameters()},
#                {'params': model.module.stage4.parameters()},
#                {'params': model.module.final_layer.parameters()}], lr=cfg.TRAIN.LR)
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.TRAIN.LR
            )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details



def _write_gt(img, gt, joints_vis, color=(0, 0, 255)):
    for i, key in enumerate(gt):
        if joints_vis[i][0] > 0.0:
            if key[0] >= 0 and key[1] >= 0 and img.shape[1] > key[0] and img.shape[0] > key[1]:
                img[int(key[1]), int(key[0]),:] = color


def save_accuracy(save_root_folder, accuracy, images, track_ids, output=None, target=None, joints=None, joints_vis=None, additionals=None):
    for i, image_path in enumerate(images):
        path_elems = image_path.split("/")
        save_dir = Path(os.path.join(save_root_folder, os.path.splitext(os.path.join(*path_elems[-3:]))[0], str(track_ids[i].item())))
        save_dir.mkdir(exist_ok=True, parents=True)

        if output is not None:
            total_hp = make_total_hp(output[i])
            if joints is not None:
                    _write_gt(total_hp, joints[i], joints_vis[i])

            cv2.imwrite(str(save_dir / ("output_" + str(accuracy[i]) + ".png")), total_hp)
        
        if target is not None:
            total_hp = make_total_hp(target[i])
            if joints is not None:
                _write_gt(total_hp, joints[i], joints_vis[i])

            cv2.imwrite(str(save_dir / ("target.png")), total_hp)

        if additionals is not None:
            for j in range(len(additionals)):
                total_hp = make_total_hp(additionals[j][1][i])
                if joints is not None:
                    _write_gt(total_hp, joints[i], joints_vis[i])

                if len(additionals[j]) >= 3:
                    cv2.imwrite(str(save_dir / (str(additionals[j][0]) + "_" + str(additionals[j][2][i]) + ".png")), total_hp)
                else:
                    cv2.imwrite(str(save_dir / (str(additionals[j][0]) + ".png")), total_hp)



def make_total_hp(heatmap):
    total = np.zeros(heatmap[0].shape, dtype=heatmap.dtype)

    for j in range(heatmap.shape[0]):
        total += heatmap[j]
        # img = heatmap[j].copy()
        # img = np.clip(img * 255, 0, 255).astype(np.uint8)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
    
    total = np.clip(total * 255, 0, 255).astype(np.uint8)
    total = cv2.cvtColor(total, cv2.COLOR_GRAY2RGB)
    return total