# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch.utils.data
import torchvision

#from .coco import build as build_coco
from .hico import build as build_hico
from .hico_gt import build as build_hico_gt
from .hico_gt_25 import build as build_hico_gt_25
from .hico_gt_nipairs import build as build_hico_gt_nipairs
from .vcoco import build as build_vcoco
from .vcoco_gt import build as build_vcoco_gt
from .vcoco_one_hot import build as build_vcoco_one_hot

from .hoia import build as build_hoia
from .hoia_gt import build as build_hoia_gt
from .hoia_one_hot import build as build_hoia_one_hot
from .hoia_one_hot_gt import build as build_hoia_one_hot_gt
from .hoia_2021 import build as build_hoia_2021
from .hoia_2021_test import build as build_hoia_2021_test


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    if args.dataset_file == 'hico_gt':
        return build_hico_gt(image_set, args)
    if args.dataset_file == 'hico_gt_25':
        return build_hico_gt_25(image_set, args)
    if args.dataset_file == 'hico_gt_nipairs':
        return build_hico_gt_nipairs(image_set, args)
    if args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    if args.dataset_file == 'vcoco_gt':
        return build_vcoco_gt(image_set, args)
    if args.dataset_file == 'vcoco_one_hot':
        return build_vcoco_one_hot(image_set, args)
    if args.dataset_file == 'hoia':
        return build_hoia(image_set, args)
    if args.dataset_file == 'hoia_gt':
        return build_hoia_gt(image_set, args)
    if args.dataset_file == 'hoia_one_hoi':
        return build_hoia_one_hot(image_set, args)
    if args.dataset_file == 'hoia_one_hoi_gt':
        return build_hoia_one_hot_gt(image_set, args)
    if args.dataset_file == 'hoia_2021':
        return build_hoia_2021(image_set, args)
    if args.dataset_file == 'hoia_2021_test':
        return build_hoia_2021_test(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
