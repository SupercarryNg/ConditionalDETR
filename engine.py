# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import numpy as np

import util.misc as utils
from datetime import datetime
from datasets.open_world_eval import OWEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, writer, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'image_id'} for t in targets]

        outputs = model(samples)
        img_features, pos = model.backbone(samples)
        # TODO img_features should be sigmoid
        loss_dict = criterion(outputs, targets, img_features[0].tensors)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # write tensorboard
        step = epoch * len(data_loader) + idx
        for loss_key, loss_value in loss_dict.items():
            writer.add_scalar(loss_key, loss_value, step)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, postprocessors, data_loader, base_ds, device):
    model.eval()
    # criterion.eval()

    print('Start Testing')

    voc_evaluator = OWEvaluator(base_ds)

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = {k: v.to(device) if k != 'image_id' else v for k, v in targets[0].items()}

        outputs = model(samples)

        orig_target_sizes = torch.stack([targets["orig_size"]], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {targets['image_id']: results}

        if voc_evaluator is not None:
            voc_evaluator.update(res)

    if voc_evaluator is not None:
        voc_evaluator.accumulate()
        voc_evaluator.summarize()


@torch.no_grad()
def visualization(model, postprocessors, data_loader, device, sample_ratio=0.1):
    print('Starting Visualization')
    model.eval()
    output_dir = 'Results/visuals/{}/'.format(datetime.now().strftime("%Y%m%d_%H%M"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sample_size = int(len(data_loader) * sample_ratio)
    sample_indices = np.random.choice(np.arange(len(data_loader)), size=sample_size)

    for idx, (samples, targets) in enumerate(data_loader):
        if idx in sample_indices:
            samples = samples.to(device)
            targets = {k: v.to(device) if k != 'image_id' else v for k, v in targets[0].items()}
            top_k = len(targets['boxes'])

            outputs = model(samples)

            indices = outputs['pred_logits'][0].softmax(-1)[..., 1].sort(descending=True)[1][:top_k]

            orig_target_sizes = torch.stack([targets["orig_size"]], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            img = read_image(os.path.join('voc_data/images/test/', targets.get('image_id')))
            for result in results:
                img = draw_bounding_boxes(img, result.get('boxes').to('cpu').unsqueeze(0),
                                          labels=[str(result.get('labels').item())])
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(output_dir, targets.get('image_id')))

        else:
            continue








