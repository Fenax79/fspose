# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
import scipy.io
import torch.nn as nn
import h5py

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from utils.utils import save_accuracy

import json
import cv2 as cv
from pathlib import Path
import json

import torch.nn.functional as F
import csv 

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    N = min(len(train_loader),config['MODEL']['ITER'])

    for i, (input,  target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:

            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        speed=input.size(0)/batch_time.val,
                        data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input, meta, target, pred*4, output,
            #                    prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, validate_type=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    filenames_map = {}
    filenames_counter = 0
    imgnums = []
    idx = 0
    end = time.time()
    
    filename_list = []

    with torch.no_grad():
        for i, (input,  target, target_weight, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)

            ########
            for ff in range(len(meta['image'])):
                cur_nm = meta['image'][ff]

                if not cur_nm in filenames_map:
                    filenames_map[cur_nm] = [filenames_counter]
                else:
                    filenames_map[cur_nm].append(filenames_counter)
                
                filename_list.append(meta['input_path'][ff])
                filenames_counter +=1
            #########

            # for only input
            if validate_type == 1:
                input_clone = torch.clone(input[:,:17])
                input_clone = input_clone.cuda(non_blocking=True)
                outputs = input_clone
            
            # compute output
            elif validate_type == 0:
                outputs = model(input)
                # first frame check
                for ff in range(len(meta['is_first_frame'])):
                    if meta['is_first_frame'][ff]:
                        outputs[ff] = input[ff][:17]
          
            # for merged cnn
            elif validate_type == 2:
                outputs = model(input)
                input_clone = torch.clone(input[:,:17])
                input_clone = input_clone.cuda(non_blocking=True)
                outputs = (outputs + input_clone) / 2
                # first frame check
                for ff in range(len(meta['is_first_frame'])):
                    if meta['is_first_frame'][ff]:
                        outputs[ff] = input[ff][:17]
    
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
            else:
                output = outputs
                loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)
            # print(avg_acc)

            #if filenames_counter == 1744:
            # if avg_acc <= 0.3:
            if False:
                _show_heatmap_result(input.detach().cpu().numpy(),
                   output.detach().cpu().numpy(), target.detach().cpu().numpy(),
                   filename_list, filenames_counter)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images


            if i % config.PRINT_FREQ == 0:

                msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time,
                            loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                # save_debug_images(config, input, meta, target, pred*4, output,
                #                    prefix)

    track_preds = None
    logger.info('########################################')
    logger.info('{}'.format(config.EXPERIMENT_NAME))
    name_values, perf_indicator = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, filenames_map, track_preds, filenames, imgnums)

    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(
            'valid_loss',
            losses.avg,
            global_steps
        )
        writer.add_scalar(
            'valid_acc',
            acc.avg,
            global_steps
        )
        if isinstance(name_values, list):
            for name_value in name_values:
                writer.add_scalars(
                    'valid',
                    dict(name_value),
                    global_steps
                )
        else:
            writer.add_scalars(
                'valid',
                dict(name_values),
                global_steps
            )
        writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def resize_and_infer(model_light, orig_input_p, orig_input_sup, large_hp_size, p_to_large):
    orig_input_p_sup = torch.cat((orig_input_p, orig_input_sup),1)
    light_input_p_sup_transferred = model_light(orig_input_p_sup)

    input_p_sup_transferred = torch.zeros(orig_input_p.shape[0], orig_input_p.shape[1],
        large_hp_size[1], large_hp_size[0],  dtype=orig_input_p.dtype)

    light_input_p_sup_transferred = light_input_p_sup_transferred.to('cpu').detach().numpy()
    light_input_p_sup_transferred = light_input_p_sup_transferred.transpose((0, 2,3,1))

    orig_input_p = orig_input_p.to('cpu').detach().numpy()
    orig_input_p = orig_input_p.transpose((0, 2,3,1))

    for i in range(orig_input_p.shape[0]):
        input_p_sup_transferred[i] = torch.from_numpy(
                cv.warpAffine(light_input_p_sup_transferred[i], 
                    np.frombuffer(p_to_large[i], dtype=np.float32).reshape((3,3))[:2],
                    (int(large_hp_size[0]), int(large_hp_size[1])),
                    flags=cv.INTER_LINEAR).transpose((2,0,1))
                )

    return input_p_sup_transferred.cuda(non_blocking=True) 


def get_accuracy(output, target, output_target_keys=False):
    # each_acc, avg_acc, cnt, pred, target_keys = 
    return accuracy(output.detach().cpu().numpy(),
        target.detach().cpu().numpy(), output_target_keys=output_target_keys)


def numpy2str(np_data):
    return " ".join(list(map(str, list(np_data.reshape(-1)))))


# TODO
# valdiate_type:0 2heatmap, 1 only input, 2 3heatnap, 3 dont use, 4: 2.5heatmap 2model,  5: merged_net
def validate_3heatmap(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, validate_type=0, save_accuracy_folder=None, write_gt=False,
             detail_info_file=None, i_frame_intreval=None, save_all_score=False, model_light=None,
             light_hp_size=None, large_hp_size=None, half=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    if half:
        model.half()

    # switch to evaluate mode
    model.eval()
    if model_light is not None:
        if half:
            model_light.half()

        model_light.eval()
    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    filenames_map = {}
    filenames_counter = 0
    imgnums = []
    idx = 0
    end = time.time()
    
    filename_list = []
    target_gt_json = {}

    if detail_info_file is not None:
        detail_info_csv = []
        detail_info_csv.append(["identity", "accuracy", "iframe distance"])

    with torch.no_grad():
        for i, (input_p, input_sup, input_i, orig_input_p, orig_input_sup, target, target_weight, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)

            for ff in range(len(meta['image'])):
                cur_nm = meta['image'][ff]

                if not cur_nm in filenames_map:
                    filenames_map[cur_nm] = [filenames_counter]
                else:
                    filenames_map[cur_nm].append(filenames_counter)
                
                filename_list.append(meta['input_path'][ff])
                filenames_counter +=1

            
            input_p = input_p.cuda(non_blocking=True) # p frame
            input_sup = input_sup.cuda(non_blocking=True) # p-1 frame
            input_i = input_i.cuda(non_blocking=True) # i frame. 


            if validate_type == 0:
                # calculate FH(t+1)
                input_sup_i = torch.cat((input_sup, input_i),1)
                #start_ = time.time()
                sup_i_transferred = model(input_sup_i)
                #end_ = time.time()
                #csv_data.append(["transfer time", str(end_-start_)])
                sup_i_merged = (sup_i_transferred + input_sup) / 2 # FH(t+1)

                # calculate FH(t+2)
                input_p_i = torch.cat((input_p, input_i),1)
                p_i_transferred = model(input_p_i)
                input_p_sup_i_merged = torch.cat((input_p, sup_i_merged),1)
                input_p_sup_i_merged_transferred = model(input_p_sup_i_merged)
                #outputs = (input_p_sup_i_merged_transferred + input_p + p_i_transferred) / 3 # FH(t+2)
                outputs = (p_i_transferred + input_p) / 2
                # from fvcore.nn.flop_count import Handle, flop_count
                # hoge = flop_count(model, (input_p_sup_i_merged.cuda(non_blocking=True), ))

                '''
                for ff in range(len(meta['is_first_frame'])):
                    if meta['is_first_frame'][ff]:
                        #print(meta['input_i_path'][ff])
                        #print(meta['image'][ff])
                        outputs[ff] = input_i[ff]
                '''
                '''
                for ff in range(len(meta['is_same_i_and_input_index'])):
                    if meta['is_same_i_and_input_index'][ff]:
                        #print(meta['input_i_path'][ff])
                        #print(meta['image'][ff])
                        outputs[ff] = input_i[ff]
                '''

                # outputs = input_i

            elif validate_type == 1:
                outputs = input_p
                #input_p_sup = torch.cat((input_p, input_sup),1)
                #p_sup_transferred = model(input_p_sup)
                #outputs = p_sup_transferred
                # outputs = outputs.cuda(non_blocking=True)
            elif validate_type == 2:

                # calculate FH(t+1)
                input_sup_i = torch.cat((input_sup, input_i),1)
                #start_ = time.time()
                sup_i_transferred = model(input_sup_i)
                #end_ = time.time()
                #csv_data.append(["transfer time", str(end_-start_)])
                sup_i_merged = (sup_i_transferred + input_sup) / 2 # FH(t+1)

                # calculate FH(t+2)
                input_p_i = torch.cat((input_p, input_i),1)
                p_i_transferred = model(input_p_i)
                input_p_sup_i_merged = torch.cat((input_p, sup_i_merged),1)
                input_p_sup_i_merged_transferred = model(input_p_sup_i_merged)
                _3hp = (input_p_sup_i_merged_transferred + input_p + p_i_transferred) / 3
                _2hp = (p_i_transferred + input_p) / 2
                outputs = _3hp

                for ff in range(len(meta['i_distance'])):
                    if int(meta['i_distance'][ff]) < i_frame_intreval / 2:
                        print(int(meta['i_distance'][ff]))
                        outputs[ff] = _2hp[ff]

            elif validate_type == 3:
                assert False
                input_p_i = torch.cat((input_p, input_i),1)
                p_i_transferred = model(input_p_i)

                input_p_sup_transferred = resize_and_infer(model_light, orig_input_p, orig_input_sup, 
                    large_hp_size, meta['p_to_large'])

                outputs = (input_p_sup_transferred + input_p + p_i_transferred) / 3 # FH(t+2)
                # outputs = (input_p + p_i_transferred) / 2
                outputs = p_i_transferred
                # outputs = input_p_sup_transferred
                # outputs = input_p # 54.201
                
            elif validate_type == 4:
                input_p_i = torch.cat((input_p, input_i),1)
                p_i_transferred = model(input_p_i)

                input_p_sup_transferred = resize_and_infer(model_light, orig_input_p, orig_input_sup, 
                    large_hp_size, meta['p_to_large'])

                outputs = (input_p_sup_transferred +  input_p + p_i_transferred) / 3

                for ff in range(len(meta['i_distance'])):
                    if int(meta['i_distance'][ff]) < i_frame_intreval / 2:
                        # print(int(meta['i_distance'][ff]))
                        outputs[ff] = (p_i_transferred[ff] + input_p[ff]) / 2

            elif validate_type == 5:
                input_p_sup = torch.cat((input_p, input_sup),1)
                p_sup_transferred = model(input_p_sup)
                outputs = (p_sup_transferred + input_p) / 2

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
            else:
                output = outputs
                loss = criterion(output, target, target_weight)

            num_images = input_i.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            each_acc, avg_acc, cnt, pred, target_keys = accuracy(output.detach().cpu().numpy(),
                                                target.detach().cpu().numpy(), output_target_keys=True)
            acc.update(avg_acc, cnt)

            if save_accuracy_folder is not None:
                joints=None
                joints_vis=None
                if write_gt:
                    joints = meta["joints"].cpu().numpy()
                    joints_vis = meta["joints_vis"].cpu().numpy()



                save_accuracy(save_accuracy_folder, each_acc, meta['image'], meta["track_id"], 
                    output=output.cpu().numpy(), target=target.cpu().numpy(), joints=joints, joints_vis=joints_vis, 
                    additionals=[("input_p", input_p.cpu().numpy(), get_accuracy(input_p, target)[0]), 
                        ("input_p_sup_transferred", input_p_sup_transferred.cpu().numpy(), get_accuracy(input_p_sup_transferred, target)[0]), 
                        ("p_i_transferred", p_i_transferred.cpu().numpy(), get_accuracy(p_i_transferred, target)[0])])

#            print(avg_acc)

            if False:
                _show_heatmap_result(input_p.detach().cpu().numpy(),
                   input_i.detach().cpu().numpy(), target.detach().cpu().numpy(),
                   filename_list, filenames_counter)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            if detail_info_file is not None:
                for ff in range(len(meta['image'])):
                    path_elems = meta['image'][ff].split("/")
                    iden = os.path.join(os.path.splitext(os.path.join(*path_elems[-3:]))[0], 
                        str(meta["track_id"][ff].item()))
                    _acc = each_acc[ff]
                    i_distance = meta['i_distance'][ff].item()
                    add_data = [iden, _acc, i_distance]

                    if save_all_score:
                        # input_p_sup_transferred
                        preds, maxvals = get_final_preds(config, input_p_sup_transferred.clone().cpu().numpy(), c, s)
                        add_data.append(numpy2str(maxvals[ff]))
                        # input_p
                        preds, maxvals = get_final_preds(config, input_p.clone().cpu().numpy(), c, s)
                        add_data.append(numpy2str(maxvals[ff]))
                        # p_i_transferred
                        preds, maxvals = get_final_preds(config, p_i_transferred.clone().cpu().numpy(), c, s)
                        add_data.append(numpy2str(maxvals[ff]))

                    detail_info_csv.append(add_data)

            idx += num_images


            if i % config.PRINT_FREQ == 0:

                msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time,
                            loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                # save_debug_images(config, input, meta, target, pred*4, output,
                #                    prefix)




    if detail_info_file is not None:
        with open(detail_info_file, "w") as f:
            writer = csv.writer(f)
            writer.writerows(detail_info_csv)
        

    '''
    import csv 
    with open("transefer_time.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    '''

    track_preds = None
    logger.info('########################################')
    logger.info('{}'.format(config.EXPERIMENT_NAME))
    name_values, perf_indicator = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, filenames_map, track_preds, filenames, imgnums)

    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)

    '''
    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(
            'valid_loss',
            losses.avg,
            global_steps
        )
        writer.add_scalar(
            'valid_acc',
            acc.avg,
            global_steps
        )
        if isinstance(name_values, list):
            for name_value in name_values:
                writer.add_scalars(
                    'valid',
                    dict(name_value),
                    global_steps
                )
        else:
            writer.add_scalars(
                'valid',
                dict(name_values),
                global_steps
            )
        writer_dict['valid_global_steps'] = global_steps + 1
    '''

    return perf_indicator

#TODO
def validate_inh_hp(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, validate_type=0, save_accuracy_folder=None, write_gt=False,
             detail_info_file=None, save_all_score=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    filenames_map = {}
    filenames_counter = 0
    imgnums = []
    idx = 0
    end = time.time()
    
    filename_list = []
    overwrite_inh_hp = True

    with torch.no_grad():
        # input_p: light heatmap shaped to large heatmap shape, input_i: large heatmap 
        for i, (input_p, input_i, target, target_weight, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            outputs = torch.zeros(input_i.shape, dtype=torch.float32)

            for ff in range(len(meta['image'])):
                cur_nm = meta['image'][ff]

                if not cur_nm in filenames_map:
                    filenames_map[cur_nm] = [filenames_counter]
                else:
                    filenames_map[cur_nm].append(filenames_counter)
                
                filename_list.append(meta['input_path'][ff])
                filenames_counter +=1

                if validate_type == 0:
                    if meta['is_i_frame'][ff]:
                        outputs[ff] = input_i[ff]
                    else:
                        current_hp = input_p[ff].cuda(non_blocking=True)
                        i_hp = input_i[ff].cuda(non_blocking=True)

                        if meta['is_next_of_i_frame'][ff]:
                            model_input = torch.cat((current_hp, i_hp))
                            model_input = model_input.reshape([1] + list(model_input.shape))
                            outputs[ff] = (model(model_input) + current_hp) / 2   
                        else:
                            model_input = torch.cat((current_hp, i_hp))
                            model_input = model_input.reshape([1] + list(model_input.shape))
                            i_transferred = model(model_input)
                            prev_inh_hp = torch.load(meta['prev_inh_hp_path'][ff], map_location=torch.device('cuda'))
                            model_input = torch.cat((current_hp, prev_inh_hp))
                            model_input = model_input.reshape([1] + list(model_input.shape))
                            outputs[ff] = (model(model_input) + current_hp + i_transferred) / 3    


                        inh_hp_path = meta['inh_hp_path'][ff]

                        if overwrite_inh_hp or not os.path.exists(inh_hp_path):
                            Path(inh_hp_path).parent.mkdir(exist_ok=True, parents=True)
                            torch.save(outputs[ff], inh_hp_path)
                elif validate_type == 1:
                    outputs[ff] = input_p[ff]

            outputs = outputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
            else:
                output = outputs
                loss = criterion(output, target, target_weight)

            num_images = input_i.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            each_acc, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            #TODO    
            if save_accuracy_folder is not None:
                assert len(meta["is_i_frame"]) == 1
                joints=None
                joints_vis=None
                if write_gt:
                    joints = meta["joints"].cpu().numpy()
                    joints_vis = meta["joints_vis"].cpu().numpy()
                
                additonals = []

                if "prev_inh_hp" in locals():
                    additonals.append(("prev_inh_hp", prev_inh_hp.cpu().numpy()))

                if not meta["is_i_frame"][0]:
                    additonals.append(("i_hp", i_hp.cpu().numpy()))

                save_accuracy(save_accuracy_folder, each_acc, meta['image'], meta["track_id"], 
                    output=output.cpu().numpy(), target=target.cpu().numpy(), joints=joints, joints_vis=joints_vis, 
                    additionals=additonals)

            if False:
                _show_heatmap_result(input_p.detach().cpu().numpy(),
                   input_i.detach().cpu().numpy(), target.detach().cpu().numpy(),
                   filename_list, filenames_counter)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images


            if i % config.PRINT_FREQ == 0:

                msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time,
                            loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                # save_debug_images(config, input, meta, target, pred*4, output,
                #                    prefix)
            
            

    track_preds = None
    logger.info('########################################')
    logger.info('{}'.format(config.EXPERIMENT_NAME))
    name_values, perf_indicator = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, filenames_map, track_preds, filenames, imgnums)

    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(
            'valid_loss',
            losses.avg,
            global_steps
        )
        writer.add_scalar(
            'valid_acc',
            acc.avg,
            global_steps
        )
        if isinstance(name_values, list):
            for name_value in name_values:
                writer.add_scalars(
                    'valid',
                    dict(name_value),
                    global_steps
                )
        else:
            writer.add_scalars(
                'valid',
                dict(name_values),
                global_steps
            )
        writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator



def _show_heatmap_result(input, output, target, filename_list, filenames_counter):
    total_outputs = []
    total_inputs = []
    total_input_sups = []
    total_targets = []

    for i in range(input.shape[0]):
        for heatmap_type in [0, 2, 3]:
            if heatmap_type == 0:
                heatmap = input[i][:17]
            #elif heatmap_type == 1:
            #    heatmap = input[i][17:]
            elif heatmap_type == 2:
                heatmap = output[i]
            elif heatmap_type == 3:
                heatmap = target[i]

            num_0_heatmap = heatmap[0].copy()
            total = np.zeros(num_0_heatmap.shape, dtype=num_0_heatmap.dtype)

            for j in range(output.shape[1]):
                img = heatmap[j].copy()
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                total += heatmap[j].copy()
                # cv.imshow("img", img)
                # cv.waitKey(0)
            
            total = np.clip(total * 255, 0, 255).astype(np.uint8)
            total = cv.cvtColor(total, cv.COLOR_GRAY2RGB)

            if heatmap_type == 0:
                total_inputs.append(total)
            #elif heatmap_type == 1:
            #    total_input_sups.append(total)
            elif heatmap_type == 2:
                total_outputs.append(total)
            elif heatmap_type == 3:
                total_targets.append(total)

    save_folder  = "/home/storm/projects/PoseWarper2/PoseWarper/temp_data"
    start = filenames_counter - input.shape[0]
    
    for i in range(input.shape[0]):
        total_input = np.clip(total_inputs[i] * 255, 0, 255).astype(np.uint8)
        cv.imshow("total_input", total_inputs[i])

        total_output = np.clip(total_outputs[i] * 255, 0, 255).astype(np.uint8)
        cv.imshow("total_output", total_outputs[i])
        total_target = np.clip(total_targets[i] * 255, 0, 255).astype(np.uint8)
        cv.imshow("total_target", total_targets[i])
        cv.waitKey(0)
        path_elems = filename_list[start+i].split("/")
        dir = os.path.join(save_folder, os.path.splitext(os.path.join(*path_elems[-5:]))[0])
        Path(dir).mkdir(exist_ok=True, parents=True)

        cv.imwrite(str(Path(dir) / ("total_output.png")), total_outputs[i])
        cv.imwrite(str(Path(dir) / ("total_input.png")), total_inputs[i])
        # cv.imwrite(str(Path(dir) / ("total_input_sup.png")), total_input_sups[i])
        cv.imwrite(str(Path(dir) / ("total_target.png")), total_targets[i])
    
    pass

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
