"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import os
from pathlib import Path
import sys

repo_root_dir: Path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(repo_root_dir))

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets.folder import has_file_allowed_extension
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
import cv2

import xmltodict
import numpy as np
from collections import defaultdict

from src.common.tools import read_file

from typing import Any, List, Dict, Union, Callable, Optional, Tuple


def detection_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)

    aggregated_targets = defaultdict(list)
    for target_dict in targets:
        for key, value in target_dict.items():
            aggregated_targets[key].append(value.tolist())

    for key in aggregated_targets:
        aggregated_targets[key] = torch.tensor(aggregated_targets[key])  # type: ignore

    print(dict(aggregated_targets))

    return images, dict(aggregated_targets)


def create_dataloaders(
    image_dir: Union[str, Path],
    annot_dir: Union[str, Path],
    transform: A.Compose,
    batch_size: int,
    num_workers: int,
    image_size: int = 512,
    target_transform: Optional[Callable] = None,
):

    # Make data folders into dataset
    dataset = LegoObjDetDataset(
        image_dir=image_dir,
        annot_dir=annot_dir,
        transform=transform,
        image_size=image_size,
        target_transform=target_transform,
    )

    # Split into training and testing data (80% training, 20% testing)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_data, test_data = random_split(dataset, [train_size, test_size])

    # Make dataset into dataloader
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(train_data),
        collate_fn=detection_collate_fn,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(test_data),
        collate_fn=detection_collate_fn,
    )

    return train_dataloader, test_dataloader, dataset


# Custom Dataset class for altering the labels of the dataset
class LegoObjDetDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        annot_dir: Union[str, Path],
        transform: A.Compose,
        image_size: int,
        target_transform: Optional[Callable] = lambda x: x,
    ) -> None:

        super().__init__()

        annotations = self.get_annotations(annot_dir, extensions=".xml")
        classes, class_to_idx = self.find_classes(annotations)
        samples = self.make_dataset(
            annotations=annotations,
            image_dir=image_dir,
            class_to_idx=class_to_idx,
            extensions=".png",
        )

        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.transform = transform
        self.image_size = image_size
        self.target_transform = target_transform

        self.annotations = annotations

        self.classes = classes
        self.class_to_idx = class_to_idx

        self.samples = samples
        self.labels = [s[1]["labels"] for s in samples]
        self.bndboxes = [s[1]["bndboxes"] for s in samples]
        self.images = [s[0] for s in samples]

        if target_transform is not None:
            transformed_classes = list(
                set(sorted(target_transform(_class) for _class in self.classes))
            )
            transformed_to_idx = {
                transformed: idx for idx, transformed in enumerate(transformed_classes)
            }
            target_to_transformed = {
                self.class_to_idx[target]: transformed_to_idx[target_transform(target)]
                for target in self.classes
            }

            self.transformed_classes = transformed_classes
            self.transformed_to_idx = transformed_to_idx
            self.target_to_transformed = target_to_transformed

    def get_annotations(
        self,
        annot_dir: Union[str, Path],
        extensions: Optional[str] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,  # type: ignore
    ) -> Dict[str, Any]:
        """Gets the image annotations from the annotation files.

        Args:
            annot_dir (Union[str, Path]): Path to directory that contains annotation files.
            extensions (Optional[str], optional): Valid file extensions for annotation files. Defaults to None.
            is_valid_file (Optional[Callable[[str], bool]], optional): Callable for checking is file is valid. Defaults to None.

        Raises:
            ValueError: Both extensions and is_valid_file cannot be None or not None at the same time.
            FileNotFoundError: Couldn't find any valid annotations files in directory.

        Returns:
            Dict[str, ndarray[Dict[str, Any], dtype[Any]]]: _description_
        """

        ann_dir = os.path.expanduser(annot_dir)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time."
            )

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)

        annotations: Dict[str, Any] = {}

        for root, _, fnames in sorted(os.walk(ann_dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):  # type: ignore
                    objects = np.squeeze(
                        list(
                            xmltodict.parse(read_file(os.path.join(root, fname)))[
                                "annotations"
                            ]["object"]
                        )
                    )
                    file_name = Path(path).stem
                    annotations.update({file_name: objects})

        if not annotations:
            raise FileNotFoundError(
                f"Couldn't find any valid annotations files in directory: {ann_dir}."
            )

        return annotations

    @staticmethod
    def find_classes(annotations: Dict[str, Any]) -> Tuple[List[str], Dict[str, int]]:
        """Finds the target classes in image annotation files.

        Args:
            annotations (ndarray[ndarray[Dict[str, Any], dtype[Any]], dtype[Any]]): Dictionary that contains annotations for alle image files.

        Raises:
            FileNotFoundError: Couldn't find any classes in given annotations.

        Returns:
            Tuple[List[str], Dict[str, int]]: Tuple that contains list with classes and dictionary for converting classes to respective indexes.
        """

        classes: List[str] = sorted(
            set(target["name"] for file in annotations.values() for target in file)
        )

        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in given annotations.")

        class_to_idx: Dict[str, int] = {
            cls_name: i for i, cls_name in enumerate(classes)
        }

        return classes, class_to_idx

    def make_dataset(
        self,
        annotations: Dict[str, Any],
        image_dir: Union[str, Path],
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[str] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,  # type: ignore
    ):
        """Makes a list with all images and the corresponding targets and bounding boxes.

        Args:
            annotations (Dict[str, ndarray[Dict[str, Any], dtype[Any]]]): Dictionary that contains annotations for alle image files.
            image_dir (Union[str, Path]): Path to directory that contain image files.
            class_to_idx (Optional[Dict[str, int]], optional): Dictionary for converting classes to respective index. Defaults to None.
            extensions (Optional[str], optional): Valid file name extensions for image files. Defaults to None.
            is_valid_file (Optional[Callable[[str], bool]], optional): Callable for checking if file is valid as image file. Defaults to None.

        Raises:
            ValueError: 'class_to_index' must have at least one entry to collect any samples.
            ValueError: Both extensions and is_valid_file cannot be None or not None at the same time.
            FileNotFoundError: Couldn't find any valid image files in directory.

        Returns:
            List[Tuple[str, List[int], List[List[int]]]]: List of samples that contains an image path and its annotations
        """

        img_dir = os.path.expanduser(image_dir)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(annotations)
        elif not class_to_idx:
            raise ValueError(
                "'class_to_index' must have at least one entry to collect any samples."
            )

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time."
            )

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)

        instances = []
        for root, _, fnames in sorted(os.walk(img_dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):  # type: ignore
                    target = annotations[Path(fname).stem]
                    item: Tuple[
                        str, Dict[str, Union[List[int], List[Dict[str, int]]]]
                    ] = path, {
                        "labels": [class_to_idx[obj["name"]] for obj in target],
                        "bndboxes": [
                            {desc: int(coor) for desc, coor in obj["bndbox"].items()}
                            for obj in target
                        ],
                    }
                    instances.append(item)

        if not instances:
            raise FileNotFoundError(
                f"Couldn't find any valid image files in directory: {img_dir}."
            )

        return instances

    def get_bbox_list(self, data, img_size):

        bbox_array = []

        for box in data:
            new_box = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]

            if new_box[2] > img_size[1]:
                new_box[2] = img_size[1]

            if new_box[3] > img_size[0]:
                new_box[3] = img_size[0]

            bbox_array.append(new_box)

        return bbox_array

    def get_output_tensors(self, data_out):
        if len(data_out["bboxes"]) > 0:
            bboxes = [
                torch.tensor(box, dtype=torch.float32) / self.image_size
                for box in data_out["bboxes"]
            ]
            labels = [int(label) for label in data_out["labels"]]
        else:
            bboxes = [torch.zeros(4)]
            labels = [-1]

        return bboxes, labels

    def __getitem__(self, index: int):

        target_to_transformed = self.target_to_transformed

        img_path, target = self.samples[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox_array = self.get_bbox_list(target["bndboxes"], image.shape)

        data_out = self.transform(
            image=image, bboxes=bbox_array, labels=target["labels"]
        )
        transformed_img = data_out["image"]
        bboxes, labels = self.get_output_tensors(data_out)

        if self.target_transform is not None:
            labels = [target_to_transformed[label] for label in labels]

        new_target = dict(
            img_idx=torch.tensor([index], dtype=torch.float32),
            img_size=torch.tensor(
                [transformed_img.shape[1], transformed_img.shape[2]],
                dtype=torch.float32,
            ),
            img_scale=torch.tensor([1.0]),
            bbox=torch.stack(bboxes),
            cls=torch.tensor(labels, dtype=torch.float32),
        )

        return transformed_img, new_target

    def __len__(self):
        return len(self.samples)
