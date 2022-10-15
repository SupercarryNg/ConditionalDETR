# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.novelty_cls = True
        self.nc_class_embed = nn.Linear(hidden_dim, 1)

        self.backbone = backbone
        self.aux_loss = aux_loss

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.nc_class_embed.bias.data = torch.ones(1) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, reference = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        outputs_classes_nc = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed(hs[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            outputs_class_nc = self.nc_class_embed(hs[lvl])
            outputs_classes_nc.append(outputs_class_nc)

        outputs_coord = torch.stack(outputs_coords)
        output_class_nc = torch.stack(outputs_classes_nc)

        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_nc_logits': output_class_nc[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, output_class_nc=None)
            if self.novelty_cls:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, output_class_nc=output_class_nc)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_class_nc=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if output_class_nc is not None:
            xx = [{'pred_logits': a, 'pred_nc_logits': c, 'pred_boxes': b}
                for a, c, b in zip(outputs_class[:-1], output_class_nc[:-1], outputs_coord[:-1])]
        else:
            xx = [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return xx


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args, num_classes, matcher, weight_dict, losses, focal_alpha):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

        # self.invalid_cls_logits = invalid_cls_logits
        self.top_unk = args.top_unk  # we use 2 in voc dataset
        # self.bbox_thresh = args.bbox_thresh

        self.unk_topk_indices = None

    def loss_NC_labels(self, outputs, targets, indices, num_boxes, owod_targets, owod_indices, log=True):
        """Novelty classification loss
        target labels will contain class as 1
        owod_indices -> indices combining matched indices + psuedo labeled indices
        owod_targets -> targets combining GT targets + psuedo labeled unknown targets
        target_classes_o -> contains all 1's
        """
        assert 'pred_nc_logits' in outputs
        src_logits = outputs['pred_nc_logits']

        idx = self._get_src_permutation_idx(owod_indices)
        target_classes_o = torch.cat([torch.full_like(t["labels"][J], 0) for t, (_, J) in zip(owod_targets, owod_indices)])
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        losses = {'loss_NC': loss_ce}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, owod_targets, owod_indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        ## comment lines from 317-320 when running for oracle settings
        temp_src_logits = outputs['pred_logits'].clone()
        # temp_src_logits[:, :, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits

        # if self.unmatched_boxes:
        #     idx = self._get_src_permutation_idx(owod_indices)
        #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(owod_targets, owod_indices)])
        # else:
        #     idx = self._get_src_permutation_idx(indices)
        #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        idx = self._get_src_permutation_idx(owod_indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(owod_targets, owod_indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, owod_targets, owod_indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, owod_targets, owod_indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, owod_targets, owod_indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, owod_targets, owod_indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'NC_labels': self.loss_NC_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, owod_targets, owod_indices, **kwargs)

    # @torch.no_grad()
    # def get_topk_indices(self, img_features, outputs, indices):
    #     bz, c, h, w = img_features.shape
    #     upsample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
    #     mean_img_feat = torch.mean(img_features, 1)  # (16, 2048, 8, 8) -> (16, 1, 8, 8)
    #
    #     src_boxes = outputs.get('pred_boxes')
    #
    #     device = src_boxes.device
    #
    #     queries = torch.arange(src_boxes.shape[1])
    #
    #     unk_topk_indices = torch.empty((0, self.top_unk), dtype=torch.int64).to(device)
    #
    #     for i, (query_idx, tgt_idx) in enumerate(indices):
    #         assert query_idx.shape == tgt_idx.shape, "optimized queries and targets number have to be matched"
    #
    #         other_idx = np.setdiff1d(queries.numpy(), query_idx)
    #         upsample_src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes[i]) * \
    #                             torch.tensor([w, h, w, h], dtype=torch.float32).to(device)  # shape -> (100, 4)
    #
    #         # (8, 8) -> (1, 1, 8, 8) -> (1, 1, h, w) -> (h, w)
    #         up_img_feat = upsample(mean_img_feat[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    #
    #         # Initialize means bounding box
    #         means_bbox = torch.zeros(queries.shape[0])
    #
    #         for j in range(queries.shape[0]):
    #             if j in other_idx:
    #                 xmin, ymin, xmax, ymax = upsample_src_boxes[j, :].long()
    #                 xmin = max(xmin, 0)
    #                 ymin = max(ymin, 0)
    #                 xmax = min(xmax, w)
    #                 ymax = min(ymax, h)
    #                 means_bbox[j] = torch.mean(up_img_feat[ymin:ymax, xmin:xmax])
    #                 if torch.isnan(means_bbox[j]):
    #                     means_bbox[j] = -1e10
    #             else:
    #                 means_bbox[j] = -1e10
    #
    #         means_bbox = means_bbox.to(device)
    #         _, unmatched_topK_idx = torch.topk(means_bbox, self.top_unk)
    #         unk_topk_indices = torch.cat([unk_topk_indices, unmatched_topK_idx.unsqueeze(0)], dim=0)
    #
    #     return unk_topk_indices

    def forward(self, outputs, targets, img_features):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             img_features: features from backbone
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # indices -> [tensor: idx of output, tensor: idx of target]

        owod_targets = deepcopy(targets)
        owod_indices = deepcopy(indices)
        owod_outputs = outputs_without_aux.copy()
        owod_device = owod_outputs["pred_boxes"].device

        #############################################################################
        # looking for unknown objects
        bz, c, h, w = img_features.shape
        upsample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
        mean_img_feat = torch.mean(img_features, 1)  # (16, 2048, 8, 8) -> (16, 1, 8, 8)

        src_boxes = outputs.get('pred_boxes')

        device = src_boxes.device

        queries = torch.arange(src_boxes.shape[1])

        for i, (query_idx, tgt_idx) in enumerate(indices):
            assert query_idx.shape == tgt_idx.shape, "optimized queries and targets number have to be matched"

            other_idx = np.setdiff1d(queries.numpy(), query_idx)
            upsample_src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes[i]) * \
                                 torch.tensor([w, h, w, h], dtype=torch.float32).to(device)  # shape -> (100, 4)

            # (8, 8) -> (1, 1, 8, 8) -> (1, 1, h, w) -> (h, w)
            up_img_feat = upsample(mean_img_feat[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            # Initialize means bounding box
            means_bbox = torch.zeros(queries.shape[0])

            for j in range(queries.shape[0]):
                if j in other_idx:
                    xmin, ymin, xmax, ymax = upsample_src_boxes[j, :].long()
                    xmin = max(xmin, 0)
                    ymin = max(ymin, 0)
                    xmax = min(xmax, w)
                    ymax = min(ymax, h)
                    means_bbox[j] = torch.mean(up_img_feat[ymin:ymax, xmin:xmax])
                    if torch.isnan(means_bbox[j]):
                        means_bbox[j] = -1e10
                else:
                    means_bbox[j] = -1e10

            means_bbox = means_bbox.to(device)
            _, topk_inds = torch.topk(means_bbox, self.top_unk)

            topk_inds = topk_inds.cpu()
            unk_label = torch.as_tensor([self.num_classes - 1], device=owod_device)  # tensor([17])
            owod_targets[i]['labels'] = torch.cat(
                (owod_targets[i]['labels'], unk_label.repeat_interleave(self.top_unk)))
            owod_indices[i] = (torch.cat((owod_indices[i][0], topk_inds)), torch.cat(
                (owod_indices[i][1], (owod_targets[i]['labels'] == unk_label).nonzero(as_tuple=True)[0].cpu())))
        #############################################################################

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, owod_targets, owod_indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_boxes, owod_targets, owod_indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class OWPostProcess(nn.Module):
    def __init__(self, score_threshold=0.2, unk_score=0.2):
        super(OWPostProcess, self).__init__()
        self.score_threshold = score_threshold
        self.unk_score = unk_score

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        assert out_logits.shape[0] == 1, "Inference only support batch size == 1"

        prob = out_logits.sigmoid().squeeze(0).cpu()

        # Get known objects
        values, labels = torch.max(prob[..., :-1], dim=-1)

        indices = np.arange(values.shape[-1])
        overthr_indices = indices[values.reshape(-1) >= self.score_threshold]

        labels = labels[overthr_indices]
        scores = values[overthr_indices]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes.squeeze(0)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).squeeze(0)
        boxes = boxes * scale_fct[None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes[overthr_indices])]

        # Find Unknown Objects
        other_indices = np.setdiff1d(indices, overthr_indices)
        for other_idx, q in zip(other_indices, prob[other_indices]):
            if 1 - q[-1] >= self.unk_score and (1 - q[:-1] <= self.score_threshold).all():
                score = 1 - q[-1]
                results.append({'scores': score, 'labels': torch.tensor(-1), 'boxes': boxes[other_idx]})

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250

    if args.dataset_file == 'voc':
        num_classes = 18

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = ConditionalDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'NC_labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(args, num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
