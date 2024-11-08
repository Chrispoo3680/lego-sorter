"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
import os
from pathlib import Path
import logging


def save_model(
    model: torch.nn.Module,
    target_dir_path: Path,
    model_name: str,
    logging_file_path: Path,
):

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  -  %(name)s  -  %(levelname)s:    %(message)s",
        handlers=[logging.FileHandler(logging_file_path), logging.StreamHandler()],
    )

    # Create target directory
    os.makedirs(target_dir_path, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt")

    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    logger.info(f"  Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def model_save_version(save_dir_path: Path, save_name: str):

    files_in_dir: list[str] = os.listdir(save_dir_path)
    version = str(sum([1 for file in files_in_dir if save_name in file]))

    save_name_version: str = f"{save_name}V{version}"

    return save_name_version
