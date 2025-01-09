"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import timm.utils

from pathlib import Path
from tqdm import tqdm
import logging
from src.common import tools

from collections import OrderedDict

from typing import Dict, List, Optional, Callable, Any, Union


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: timm.utils.NativeScaler,
    device: torch.device,
):

    losses_m = timm.utils.AverageMeter()

    model.train()

    for batch, (X, y) in enumerate(
        tqdm(
            dataloader,
            position=1,
            leave=False,
            desc="Iterating through training batches.",
        )
    ):
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            X, y = X.to(device), y.to(device)
            output = model(X, y)

        loss = output["loss"]

        losses_m.update(loss.item(), X.size(0))

        optimizer.zero_grad(set_to_none=True)

        scaler(loss, optimizer)

        torch.cuda.synchronize()

    return OrderedDict([("loss", losses_m.avg)])


def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
):

    losses_m = timm.utils.AverageMeter()

    model.eval()

    with torch.autocast(device_type=device.type, dtype=torch.float16):
        with torch.inference_mode():
            for batch, (X, y) in enumerate(
                tqdm(
                    dataloader,
                    position=1,
                    leave=False,
                    desc="Iterating through testing batches.",
                )
            ):
                X, y = X.to(device), y.to(device)

                output = model(X, y)
                loss = output["loss"]

                torch.cuda.synchronize()
                losses_m.update(loss.data.item(), X.size(0))

    metrics = OrderedDict([("loss", losses_m.avg)])

    return metrics


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    lr_scheduler: Union[MultiStepLR, StepLR],
    epochs: int,
    device: torch.device,
    logging_file_path: Path,
    early_stopping: Any,
    scaler: timm.utils.NativeScaler,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, List[float]]:

    logger: logging.Logger = tools.create_logger(
        log_path=logging_file_path, logger_name=__name__
    )

    results: Dict[str, List[float]] = {
        "learning_rate": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(epochs), position=0, desc="Iterating through epochs."):
        train_metrics = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        test_metrics = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        early_stopping(test_metrics["loss"], model, epoch + 1)

        # Log and save epoch loss and accuracy results
        logger.info(
            f"      Epoch: {epoch+1}  |  "
            f"train_loss: {train_metrics['loss']:.4f}  |  "
            f"test_loss: {train_metrics['loss']:.4f}  |  "
            f"learning_rate: {optimizer.param_groups[0]['lr']}  |  "
            f"early stopping counter: {early_stopping.counter} / {early_stopping.patience}"
        )

        results["learning_rate"].append(optimizer.param_groups[0]["lr"])
        results["train_loss"].append(train_metrics["loss"])
        results["test_loss"].append(test_metrics["loss"])

        # See if there's a writer, if so, log to it
        if writer:
            writer.add_scalar(
                tag="Learning rate",
                scalar_value=optimizer.param_groups[0]["lr"],
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={
                    "train_loss": train_metrics["loss"],
                    "test_loss": test_metrics["loss"],
                },
                global_step=epoch,
            )

            writer.close()

        # Check if test loss is still decreasing. If not decreasing for multiple epochs, break the loop
        if early_stopping.early_stop:
            logger.info(
                f"Models test loss not decreasing significantly enough. Stopping training early at epoch: {epoch+1}"
            )
            logger.info(
                f"Saving model with lowest loss from epoch: {early_stopping.best_score_epoch}"
            )

            break

        # Adjust learning rate
        lr_scheduler.step()

    return results
