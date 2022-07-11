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
    elif args.model_name =='AMM':
        from models.Hard_Sample.AMM.hoi_hardm_query_att_each_pos import build
    elif args.model_name == 'HQM':
        from models.Hard_Sample.HQM.hoi_HQM import build
    else:
        assert False
        from .detr import build

    return build(args)
