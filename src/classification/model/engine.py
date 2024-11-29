"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

from pathlib import Path
from tqdm import tqdm
import logging
from src.common import tools

from typing import Dict, List, Optional


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(
        tqdm(dataloader, position=1, leave=False, desc="Iterating through batches.")
    ):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
):

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    epochs: int,
    device: torch.device,
    logging_file_path: Path,
    early_stopping,
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
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        # Adjust learning rate
        lr_scheduler.step()

        # Log and save epoch loss and accuracy results
        logger.info(
            f"      Epoch: {epoch+1}  |  "
            f"train_loss: {train_loss:.4f}  |  "
            f"train_acc: {train_acc:.4f}  |  "
            f"test_loss: {test_loss:.4f}  |  "
            f"test_acc: {test_acc:.4f}  |  "
            f"learning_rate: {optimizer.param_groups[0]['lr']}  |  "
            f"early stopping counter: {early_stopping.counter}"
        )

        results["learning_rate"].append(optimizer.param_groups[0]["lr"])
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # See if there's a writer, if so, log to it
        if writer:
            writer.add_scalar(
                tag="Learning rate",
                scalar_value=optimizer.param_groups[0]["lr"],
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                global_step=epoch,
            )

            writer.close()

        # Check if test loss is still decreasing. If not decreasing for multiple epochs, break the loop
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            logger.info(
                f"Models test loss not decreasing significantly enough. Stopping training early at epoch: {epoch+1}, and saving model with lowest loss."
            )
            break

    return results
