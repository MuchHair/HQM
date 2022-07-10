# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


def build_model(args):
    if args.model_name =='hoi_qpos_shiftbbox':
        from models.Hard_Sample.hoi_share_qpos_ezero_shiftbbox import build
    elif args.model_name =='hoi_qpos_shiftbbox_pos_neg':
        from models.Hard_Sample.hoi_share_qpos_ezero_shift_bbox_pos_neg import build
    elif args.model_name =='hoi_qpos_shiftbbox_gt_neg':
        from models.Hard_Sample.hoi_share_qpos_ezero_shift_bbox_gt_neg import build
    elif args.model_name =='hoi_qpos_shiftbbox_gt_mAP':
        from models.Hard_Sample.hoi_share_qpos_ezero_shiftbbox_gt_mAP import build
    elif args.model_name =='hoi_qpos_shiftbbox_gt_mAP_ts':
        from models.Hard_Sample.hoi_share_qpos_ezero_shiftbbox_04_06_ts import build
    else:
        from .detr import build

    return build(args)
