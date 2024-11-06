"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
import os
from pathlib import Path
import logging


def save_model(model: torch.nn.Module, target_dir_path: Path, model_name: str):

    logging.basicConfig(level=logging.INFO)

    # Create target directory
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt")

    # Checks if there are already files with the same name in target directory
    files_in_dir: list[str] = os.listdir(target_dir_path)
    version = str(sum([1 for file in files_in_dir if model_name in file]))

    file_name, file_type = model_name.split(".")
    save_name: str = f"{file_name}V{version}.{file_type}"

    model_save_path: Path = target_dir_path / save_name

    # Save the model state_dict()
    logging.info("  Saving model to:    ", model_save_path)
    torch.save(obj=model.state_dict(), f=model_save_path)
