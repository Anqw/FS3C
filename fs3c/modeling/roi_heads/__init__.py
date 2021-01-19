# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .box_head import ROI_BOX_HEAD_REGISTRY, build_box_head
from .roi_heads import (
    ROI_HEADS_REGISTRY,
    ROIHeads,
    StandardROIHeads,
    build_roi_heads,
    build_jig_heads,
    build_rot_heads,
    select_foreground_proposals,
)
