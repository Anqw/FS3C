# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pkg_resources
import torch

from fs3c.checkpoint import DetectionCheckpointer
from fs3c.config import get_cfg
from fs3c.modeling import build_model


class _ModelZooUrls(object):
    """
    Mapping from names to our pre-trained models.
    """
    URL_PREFIX = " "

    # format: {config_path.yaml} -> model_id/model_final.pth
    CONFIG_PATH_TO_URL_SUFFIX = {
        ### PASCAL VOC Detection ###
        # Base Model
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml": "voc/split1/base_model/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_base2.yaml": "voc/split2/base_model/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_base3.yaml": "voc/split3/base_model/model_final.pth",

        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml": "voc/split1/fs3c_1shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot.yaml": "voc/split1/fs3c_2shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_3shot.yaml": "voc/split1/fs3c_3shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml": "voc/split1/fs3c_5shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml": "voc/split1/fs3c_10shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_1shot.yaml": "voc/split2/fs3c_1shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_2shot.yaml": "voc/split2/fs3c_2shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_3shot.yaml": "voc/split2/fs3c_3shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_5shot.yaml": "voc/split2/fs3c_5shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot.yaml": "voc/split2/fs3c_10shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_1shot.yaml": "voc/split3/fs3c_1shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_2shot.yaml": "voc/split3/fs3c_2shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_3shot.yaml": "voc/split3/fs3c_3shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_5shot.yaml": "voc/split3/fs3c_5shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot.yaml": "voc/split3/fs3c_10shot/model_final.pth",

        ### COCO Detection ###
        # Base Model
        "COCO-detection/faster_rcnn_R_101_FPN_base.yaml": "coco/base_model/model_final.pth",

        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml": "coco/fs3c_1shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_2shot.yaml": "coco/fs3c_2shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_3shot.yaml": "coco/fs3c_3shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_5shot.yaml": "coco/fs3c_5shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot.yaml": "coco/fs3c_10shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml": "coco/fs3c_30shot/model_final.pth",
    }


def get_checkpoint_url(config_path):
    """
    Returns the URL to the model trained using the given config
    Args:
        config_path (str): config file name relative to Fs3c's "configs/"
            directory, e.g., "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
    Returns:
        str: a URL to the model
    """
    if config_path in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
        suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[config_path]
        return _ModelZooUrls.URL_PREFIX + suffix
    raise RuntimeError("{} not available in Model Zoo!".format(name))


def get_config_file(config_path):
    """
    Returns path to a builtin config file.
    Args:
        config_path (str): config file name relative to Fs3c's "configs/"
            directory, e.g., "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename(
        "fs3c.model_zoo", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file


def get(config_path, trained: bool = False):
    """
    Get a model specified by relative path under Fs3c's official ``configs/`` directory.
    Args:
        config_path (str): config file name relative to Fs3c's "configs/"
            directory, e.g., "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
        trained (bool): If True, will initialize the model with the trained model zoo weights.
            If False, the checkpoint specified in the config file's ``MODEL.WEIGHTS`` is used
            instead; this will typically (though not always) initialize a subset of weights using
            an ImageNet pre-trained model, while randomly initializing the other weights.
    Example:
    .. code-block:: python
        from fs3c import model_zoo
        model = model_zoo.get("COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml", trained=True)
    """
    cfg_file = get_config_file(config_path)

    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    if trained:
        cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model
