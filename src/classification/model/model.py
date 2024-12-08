"""
Contains PyTorch model code to instantiate a 'EfficientNet_B0' model.
"""

import torch
from torch import nn
from torchvision import models

import timm
import timm.data

from typing import List, Union


def get_timm_model(
    model_name: str,
    num_classes: int,
    device: torch.device,
    pretrained: bool = False,
    checkpoint_path: str = "",
    frozen_blocks: List[int] = [],
):

    if model_name not in timm.list_models(""):
        raise KeyError(
            "Model name is not valid! Must be a model from timm.list_models('*efficientnet*'). "
        )

    # Getting model architecture from timm
    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
    ).to(device)

    # Freeze given blocks in the 'features' section of the model
    for idx in frozen_blocks:
        if idx not in range(len(model.blocks)):
            break
        for param in model.blocks[idx].parameters():
            param.requires_grad = False

    img_transform = timm.data.transforms_factory.create_transform(
        **timm.data.resolve_data_config(model.pretrained_cfg, model=model)
    )

    return model, img_transform


def get_tv_efficientnet_b0(
    num_classes: int,
    device: torch.device,
    pretrained: bool = False,
    checkpoint_path: str = "",
    frozen_blocks: List[int] = [],
):

    # Get the default weights for 'efficientnet_b0'
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        transform = weights.transforms()
    else:
        weights = None
        transform = None

    # Transfering the model 'efficientnet_b0'
    efficientnet_b0 = models.efficientnet_b0(weights=weights).to(device)

    if checkpoint_path:
        efficientnet_b0.load_state_dict(torch.load(checkpoint_path), strict=False)

    # Freeze given blocks in the 'features' section of the model
    for idx in frozen_blocks:
        if idx not in range(len(efficientnet_b0.features)):
            break
        for param in efficientnet_b0.features[idx].parameters():
            param.requires_grad = False

    # Recreate the classifier layer and seed it to the target device
    efficientnet_b0.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(
            in_features=1280,
            out_features=num_classes,
            bias=True,
        ),
    ).to(device)

    return efficientnet_b0, transform
