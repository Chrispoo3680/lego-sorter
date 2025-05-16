"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from pathlib import Path
import os
from tqdm import tqdm
from src.common import tools

from typing import Dict, List, Optional, Callable, Any, Union, Tuple


try:
    logging_file_path = os.environ["LOGGING_FILE_PATH"]
except KeyError:
    logging_file_path = None

logger = tools.create_logger(log_path=logging_file_path, logger_name=__name__)


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
):

    model.train()

    train_loss, train_acc = 0, 0

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
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        optimizer.zero_grad(set_to_none=True)

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
    lr_scheduler: Union[MultiStepLR, StepLR],
    epochs: int,
    device: torch.device,
    temp_checkpoint_file_path: Path,
    early_stopping: Any,
    scaler: GradScaler,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:

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
            scaler=scaler,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        early_stopping(test_loss, model, epoch + 1)

        # Log and save epoch loss and accuracy results
        logger.info(
            f"      Epoch: {epoch+1}  |  "
            f"train_loss: {train_loss:.4f}  |  "
            f"train_acc: {train_acc:.4f}  |  "
            f"test_loss: {test_loss:.4f}  |  "
            f"test_acc: {test_acc:.4f}  |  "
            f"learning_rate: {optimizer.param_groups[0]['lr']}  |  "
            f"early stopping counter: {early_stopping.counter} / {early_stopping.patience}"
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
        if early_stopping.early_stop:

            logger.info(
                f"Models test loss not decreasing significantly enough. Stopping training early at epoch: {epoch+1}"
            )
            logger.info(
                f"Saving model with lowest loss from epoch: {early_stopping.best_score_epoch}"
            )

            os.remove(temp_checkpoint_file_path)

            break

        else:
            torch.save(obj=early_stopping.best_model_state, f=temp_checkpoint_file_path)

        # Adjust learning rate
        lr_scheduler.step()

    return results, early_stopping.best_model_state
