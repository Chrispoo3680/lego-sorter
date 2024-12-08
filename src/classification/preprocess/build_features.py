"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

import os
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset

from typing import Any, List, Dict, Union, Callable, Optional


NUM_WORKERS: int = 0 if os.cpu_count() is None else os.cpu_count()  # type: ignore


def create_dataloaders(
    data_dir_path: List[Path],
    batch_size: int,
    transform: Union[transforms.Compose, Any] = None,
    target_transform: Union[Optional[Callable], None] = None,
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

    train_data, test_data = random_split(
        full_dataset, [train_size, test_size], torch.manual_seed(0)
    )

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


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


# Custom Dataset class for altering the labels of the dataset
class PartSortingDataset(datasets.DatasetFolder):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )
        self.imgs = self.samples

        if self.target_transform is not None:
            transformed_classes = set(
                sorted(self.target_transform(_class) for _class in self.classes)
            )
            transformed_to_idx = {
                transformed: idx for idx, transformed in enumerate(transformed_classes)
            }
            target_to_transformed = {
                self.class_to_idx[target]: transformed_to_idx[
                    self.target_transform(target)
                ]
                for target in self.classes
            }

            self.transformed_classes = transformed_classes
            self.transformed_to_idx = transformed_to_idx
            self.target_to_transformed = target_to_transformed

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target_to_transformed = self.target_to_transformed
            target = target_to_transformed[target]

        return sample, target

    def __len__(self) -> int:
        return len(self.imgs)


# Dataset class for turning subsets into datasets with transforms
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        sample, target = self.subset[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.subset)
