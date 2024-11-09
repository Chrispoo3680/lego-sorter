"""
Contains PyTorch model code to instantiate a 'EfficientNet_B0' model.
"""

import torch
from torch import nn
from torchvision import models


def create_efficientnet_b0(class_names: list[str], device: torch.device):

    # Get the default weights for 'efficientnet_b0'
    weights = models.EfficientNet_B0_Weights.DEFAULT

    # Transfering the model 'efficientnet_b0'
    efficientnet_b0 = models.efficientnet_b0(weights=weights).to(device)

    # Freeze all base layers in the 'features' section of the model
    for param in efficientnet_b0.features.parameters():
        param.requires_grad = False

    unfrozen_blocks: list[int] = [7, 8]

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


def create_efficientnet_b1(class_names: list[str], device: torch.device):

    # Get the default weights for 'efficientnet_b1'
    weights = models.EfficientNet_B1_Weights.DEFAULT

    # Transfering the model 'efficientnet_b1'
    efficientnet_b1 = models.efficientnet_b1(weights=weights).to(device)

    # Freeze all base layers in the 'features' section of the model
    for param in efficientnet_b1.features.parameters():
        param.requires_grad = False

    unfrozen_blocks: list[int] = [7, 8]

    for block in unfrozen_blocks:
        for param in efficientnet_b1.features[block].parameters():
            param.requires_grad = True

    # Get the length of class_names (one output unit for each class)
    output_shape: int = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    efficientnet_b1.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(
            in_features=1280,
            out_features=output_shape,
            bias=True,
        ),
    ).to(device)

    return efficientnet_b1, weights


def create_efficientnet_b3(class_names: list[str], device: torch.device):

    # Get the default weights for 'efficientnet_b3'
    weights = models.EfficientNet_B3_Weights.DEFAULT

    # Transfering the model 'efficientnet_b3'
    efficientnet_b3 = models.efficientnet_b3(weights=weights).to(device)

    # Freeze all base layers in the 'features' section of the model
    for param in efficientnet_b3.features.parameters():
        param.requires_grad = False

    unfrozen_blocks: list[int] = [8]

    for block in unfrozen_blocks:
        for param in efficientnet_b3.features[block].parameters():
            param.requires_grad = True

    # Get the length of class_names (one output unit for each class)
    output_shape: int = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    efficientnet_b3.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(
            in_features=1536,
            out_features=output_shape,
            bias=True,
        ),
    ).to(device)

    return efficientnet_b3, weights
