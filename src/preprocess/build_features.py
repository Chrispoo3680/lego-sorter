"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

import os
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


NUM_WORKERS: int = 0 if os.cpu_count() is None else os.cpu_count()  # type: ignore


def create_dataloaders(
    data_dir_path: Path,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):

    # Make data folder into dataset
    full_dataset = datasets.ImageFolder(
        root=data_dir_path, transform=transform, target_transform=None
    )

    # Split into training and testing data (80% training, 20% testing)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_data, test_data = random_split(full_dataset, [train_size, test_size])

    # Make dataset into dataloader
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader
