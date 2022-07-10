# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


def build_model(args):
    if args.model_name =='GBS':
        from models.Hard_Sample.GBS.hoi_share_qpos_ezero_shiftbbox_04_06 import build
    elif args.model_name =='GBS_DN':
        from models.Hard_Sample.GBS.DN_DETR import build
    elif args.model_name == 'HQM':
        from models.Hard_Sample.Fuse.hoi_HQM import build
    elif args.model_name == 'GBS_TS':
        from models.TS.DOQ import build
    elif args.model_name == 'HQM_weight':
        from models.Hard_Sample.Fuse.hoi_HQM_weight import build
    else:
        from .detr import build

    return build(args)
