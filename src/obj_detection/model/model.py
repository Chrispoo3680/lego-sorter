"""
Contains PyTorch model code to instantiate a 'EfficientNet_B0' model.
"""

import torch
from torch import nn
from torchvision import models

import timm
import timm.data

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union


def timm_create_model(
    model_name: str,
    class_names: List[str],
    device: torch.device,
    pretrained: bool = False,
    frozen_blocks: List[int] = [],
):

    # Getting model architecture from timm
    model = timm.create_model(model_name=model_name, pretrained=pretrained).to(device)

    # Freeze given blocks in the 'features' section of the model
    for idx in frozen_blocks:
        for param in model.blocks[idx].parameters():
            param.requires_grad = False

    # Get the output and input shapes for the classifier
    output_shape: int = len(class_names)
    input_shape = model.classifier.in_features

    # Recreate the classifier layer and seed it to the target device
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(
            in_features=input_shape,
            out_features=output_shape,
            bias=True,
        ),
    ).to(device)

    img_transform = timm.data.transforms_factory.create_transform(
        **timm.data.resolve_data_config(model.pretrained_cfg, model=model)
    )

    return model, img_transform


def create_efficientnet_b0(class_names: List[str], device: torch.device):

    # Get the default weights for 'efficientnet_b0'
    weights = models.EfficientNet_B0_Weights.DEFAULT

    # Transfering the model 'efficientnet_b0'
    efficientnet_b0 = models.efficientnet_b0(weights=weights).to(device)

    # Freeze all base layers in the 'features' section of the model
    for param in efficientnet_b0.features.parameters():
        param.requires_grad = False

    unfrozen_blocks: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    for block in unfrozen_blocks:
        for param in efficientnet_b0.features[block].parameters():
            param.requires_grad = True

    # Get the length of class_names (one output unit for each class)
    output_shape: int = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    efficientnet_b0.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(
            in_features=1280,
            out_features=output_shape,
            bias=True,
        ),
    ).to(device)

    return efficientnet_b0, weights