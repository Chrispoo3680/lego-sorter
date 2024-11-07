"""
Contains PyTorch model code to instantiate a 'EfficientNet_B0' model.
"""

import torch
from torchvision import models


def get_model_efficientnet_b0(class_names: list[str], device: torch.device):

    # Get the default weights for 'efficientnet_b0'
    weights = models.EfficientNet_B0_Weights.DEFAULT

    # Transfering the model 'efficientnet_b0'
    efficientnet_b0 = models.efficientnet_b0(weights=weights).to(device)

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in efficientnet_b0.features.parameters():
        param.requires_grad = False

    # Get the length of class_names (one output unit for each class)
    output_shape: int = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    efficientnet_b0.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=output_shape,
            bias=True,
        ),
    ).to(device)

    return efficientnet_b0, weights


def get_model_efficientnet_b1(class_names: list[str], device: torch.device):

    # Get the default weights for 'efficientnet_b1'
    weights = models.EfficientNet_B1_Weights.DEFAULT

    # Transfering the model 'efficientnet_b1'
    efficientnet_b1 = models.efficientnet_b1(weights=weights).to(device)

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in efficientnet_b1.features.parameters():
        param.requires_grad = False

    """
    for param in efficientnet_b1.features[8].parameters():
        param.requires_grad = True
    """

    # Get the length of class_names (one output unit for each class)
    output_shape: int = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    efficientnet_b1.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=output_shape,
            bias=True,
        ),
    ).to(device)

    return efficientnet_b1, weights


def get_model_efficientnet_b3(class_names: list[str], device: torch.device):

    # Get the default weights for 'efficientnet_b3'
    weights = models.EfficientNet_B3_Weights.DEFAULT

    # Transfering the model 'efficientnet_b3'
    efficientnet_b3 = models.efficientnet_b3(weights=weights).to(device)

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in efficientnet_b3.features.parameters():
        param.requires_grad = False

    """
    for param in efficientnet_b3.features[8].parameters():
        param.requires_grad = True
    """

    # Get the length of class_names (one output unit for each class)
    output_shape: int = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    efficientnet_b3.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1536,
            out_features=output_shape,
            bias=True,
        ),
    ).to(device)

    return efficientnet_b3, weights
