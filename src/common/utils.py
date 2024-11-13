"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
from torch.utils.tensorboard.writer import SummaryWriter

import os
from pathlib import Path
import logging
from src.common import tools


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        # If first validation loss, set it as best_score
        if self.best_score is None:
            self.best_score = val_loss
        # Check if there’s an improvement
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0  # Reset patience counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def save_model(
    model: torch.nn.Module,
    target_dir_path: Path,
    model_name: str,
    logging_file_path: Path,
):

    logger: logging.Logger = tools.create_logger(
        log_path=logging_file_path, logger_name=__name__
    )

    # Create target directory
    os.makedirs(target_dir_path, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt")

    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    logger.info(f"  Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def model_save_version(save_dir_path: Path, save_name: str) -> str:

    files_in_dir: list[str] = os.listdir(save_dir_path)
    version = str(sum([1 for file in files_in_dir if save_name in file]))

    save_name_version: str = f"{save_name}V{version}"

    return save_name_version


def create_writer(
    root_dir: Path,
    experiment_name: str,
    model_name: str,
    var: str,
    logging_file_path: Path,
) -> SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    Args:
        root_dir (Path): Root dir of the repository.
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        var (str): The varying factor to change when experimenting on what gives the best results. Also called the 'independent variable'.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
    """

    log_dir: str = os.path.join(root_dir, "runs", experiment_name, model_name, var)

    logger: logging.Logger = tools.create_logger(
        log_path=logging_file_path, logger_name=__name__
    )
    logger.info(f"Created SummaryWriter, saving to:  {log_dir}...")

    return SummaryWriter(log_dir=log_dir)
