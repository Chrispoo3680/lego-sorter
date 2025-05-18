import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP

from timm.utils.cuda import NativeScaler
from timm.utils.metrics import AverageMeter

from pathlib import Path
import os
from tqdm import tqdm
from collections import OrderedDict

from ...common import tools, utils


from typing import Dict, List, Optional, Any, Union, Tuple


try:
    logging_file_path = os.environ["LOGGING_FILE_PATH"]
except KeyError:
    logging_file_path = None

logger = tools.create_logger(log_path=logging_file_path, logger_name=__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: torch.device,
        rank: int,
        scaler: NativeScaler,
        early_stopping: utils.EarlyStopping,
        lr_scheduler: Optional[Union[MultiStepLR, StepLR]] = None,
        temp_checkpoint_file_path: Optional[Path] = None,
        writer: Optional[SummaryWriter] = None,
    ):

        self.device = device
        self.rank = rank
        self.model = DDP(model.to(rank), device_ids=[rank])
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.temp_checkpoint_file_path = temp_checkpoint_file_path
        self.writer = writer

    def train_step(self, epoch: int):

        self.train_dataloader.sampler.set_epoch(epoch)  # type: ignore

        losses_m = AverageMeter()

        self.model.train()

        for batch, (X, y) in enumerate(
            tqdm(
                self.train_dataloader,
                position=1,
                leave=False,
                desc="Iterating through training batches.",
                disable=self.rank != 0,
            )
        ):
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                X = X.to(self.rank)
                y = {k: v.to(self.rank) for k, v in y.items()}
                output = self.model(X, y)

            loss = output["loss"]

            losses_m.update(loss.item(), X.size(0))

            self.optimizer.zero_grad(set_to_none=True)

            self.scaler(loss, self.optimizer)

            torch.cuda.synchronize()

        return OrderedDict([("loss", losses_m.avg)])

    def test_step(self, epoch: int):

        self.test_dataloader.sampler.set_epoch(epoch)  # type: ignore

        losses_m = AverageMeter()

        self.model.eval()

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            with torch.inference_mode():
                for batch, (X, y) in enumerate(
                    tqdm(
                        self.test_dataloader,
                        position=1,
                        leave=False,
                        desc="Iterating through testing batches.",
                        disable=self.rank != 0,
                    )
                ):
                    X = X.to(self.rank)
                    y = [{k: v.to(self.rank) for k, v in t.items()} for t in y]

                    output = self.model(X, y)
                    loss = output["loss"]

                    torch.cuda.synchronize()
                    losses_m.update(loss.data.item(), X.size(0))

        return OrderedDict([("loss", losses_m.avg)])

    def train(self, epochs: int) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:

        results: Dict[str, List[float]] = {
            "learning_rate": [],
            "train_loss": [],
            "test_loss": [],
        }

        for epoch in tqdm(
            range(epochs),
            position=0,
            desc="Iterating through epochs.",
            disable=self.rank != 0,
        ):
            train_metrics = self.train_step(epoch)
            test_metrics = self.test_step(epoch)

            results["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            results["train_loss"].append(train_metrics["loss"])
            results["test_loss"].append(test_metrics["loss"])

            self.early_stopping(test_metrics["loss"], self.model.module, epoch + 1)

            if self.rank == 0:

                early_stop_flag = int(self.early_stopping.early_stop)

                # Log and save epoch loss and accuracy results
                logger.info(
                    f"      GPU ID: {self.rank}  |  "
                    f"epoch: {epoch+1}  |  "
                    f"train_loss: {train_metrics['loss']:.4f}  |  "
                    f"test_loss: {train_metrics['loss']:.4f}  |  "
                    f"learning_rate: {self.optimizer.param_groups[0]['lr']}  |  "
                    f"early stopping counter: {self.early_stopping.counter} / {self.early_stopping.patience}"
                )

                # See if there's a writer, if so, log to it
                if self.writer:
                    self.writer.add_scalar(
                        tag="Learning rate",
                        scalar_value=self.optimizer.param_groups[0]["lr"],
                        global_step=epoch,
                    )
                    self.writer.add_scalars(
                        main_tag="Loss",
                        tag_scalar_dict={
                            "train_loss": train_metrics["loss"],
                            "test_loss": test_metrics["loss"],
                        },
                        global_step=epoch,
                    )

                    self.writer.close()

            else:
                early_stop_flag = 0

            # Sync across all ranks
            should_stop_tensor = torch.tensor(early_stop_flag, device=self.rank)
            torch.distributed.broadcast(should_stop_tensor, src=0)
            should_stop = bool(should_stop_tensor.item())

            # Check if test loss is still decreasing. If not decreasing for multiple epochs, break the loop
            if should_stop:
                if self.rank == 0:

                    logger.info(
                        f"Models test loss not decreasing significantly enough. Stopping training early at epoch: {epoch+1}"
                    )
                    logger.info(
                        f"Saving model with lowest loss from epoch: {self.early_stopping.best_score_epoch}"
                    )

                    if self.temp_checkpoint_file_path is not None:
                        os.remove(self.temp_checkpoint_file_path)

                break

            elif self.rank == 0 and self.temp_checkpoint_file_path is not None:
                torch.save(
                    obj=self.early_stopping.best_model_state,
                    f=self.temp_checkpoint_file_path,
                )

            # Adjust learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return results, self.early_stopping.best_model_state
