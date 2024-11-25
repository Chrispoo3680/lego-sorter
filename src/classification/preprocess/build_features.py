"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

import os
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset

from typing import Any, List, Union, Callable, Optional


NUM_WORKERS: int = 0 if os.cpu_count() is None else os.cpu_count()  # type: ignore


def create_dataloaders(
    data_dir_path: List[Path],
    transform: Union[transforms.Compose, Any],
    target_transform: Optional[Callable],
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):

    # Make data folders into dataset
    independent_datasets: List[PartSortingDataset] = []
    for path in data_dir_path:
        independent_datasets.append(
            PartSortingDataset(
                root=path, transform=transform, target_transform=target_transform
            )
        )

    full_dataset = ConcatDataset(independent_datasets)

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


# Custom Dataset class for altering the labels of the dataset
class PartSortingDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        class_to_idx = self.class_to_idx

        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target, class_to_idx)

        return sample, target
