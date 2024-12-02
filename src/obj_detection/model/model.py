"""
Contains PyTorch model code to instantiate a 'EfficientNet_B0' model.
"""

import torch
from torch import nn
from torchvision import models

import timm
import timm.data
import effdet

from pathlib import Path

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union


def effdet_create_model(
    model_name: str,
    num_classes: int,
    device: torch.device,
    pretrained_backbone: bool = True,
    image_size: int = 512,
    backbone_checkpoint_path: Union[str, Path] = "",
    frozen_backbone: bool = False,
    max_det_per_image: int = 100,
):

    possible_models = list(
        effdet.config.model_config.efficientdet_model_param_dict.keys()
    )

    if model_name not in possible_models:
        raise KeyError(
            "Model name is not valid! Must be a model from the effdet model config list. "
        )

    config = effdet.get_efficientdet_config(model_name)

    config.update({"num_classes": num_classes})
    config.update({"image_size": [image_size, image_size]})
    config.update({"max_det_per_image": max_det_per_image})

    # Getting model architecture
    net = effdet.EfficientDet(config, pretrained_backbone=pretrained_backbone).to(
        device
    )
    net.class_net = effdet.efficientdet.HeadNet(
        config,
        num_outputs=config.num_classes,
    )

    if backbone_checkpoint_path:
        net.backbone.load_state_dict(torch.load(backbone_checkpoint_path), strict=False)

    if frozen_backbone:
        for param in net.backbone.parameters():
            param.requires_grad = False

    return net, config
