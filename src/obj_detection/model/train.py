"""
This is a file for training the lego object detection model.
"""

import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


import timm.utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import cv2

import sys
from pathlib import Path

repo_root_dir: Path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(repo_root_dir))

import os
import logging
import argparse
import json

from src.data import download
from src.common import utils, tools
from src.obj_detection.preprocess import build_features
import engine, obj_detection.model.models as models

from typing import Dict, List


# Setup arguments parsing for hyperparameters
parser = argparse.ArgumentParser(description="Hyperparameter configuration")

parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
parser.add_argument(
    "--frozen_backbone",
    type=bool,
    default=True,
    help="Number of blocks to be frozen on format 'block1,block2,block3'",
)
parser.add_argument(
    "--lr_step_interval",
    type=int,
    default=10,
    help="Step interval for the learning rate",
)
parser.add_argument(
    "--unfrozen_backbone_blocks",
    type=str,
    default="",
    help="Number of blocks to be unfrozen on format 'block1,block2,block3'",
)
parser.add_argument(
    "--pretrained_backbone",
    type=bool,
    default=False,
    help="If model should use pretrained backbone weights",
)
parser.add_argument(
    "--backbone_path",
    type=str,
    default="",
    help="Path to checkpoint used to initialize model backbone weights",
)
parser.add_argument(
    "--target_transform",
    type=bool,
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Target transform is applied",
)
parser.add_argument("--image_size", type=int, default=1024, help="Image size")
parser.add_argument("--model_name", type=str, required=True, help="Loaded models name")
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
parser.add_argument(
    "--experiment_variable", type=str, default=None, help="Experiment variable"
)

args, _ = parser.parse_known_args()

parser.add_argument(
    "--model_save_name",
    type=str,
    default=args.model_name + "_lego_objdet",
    help="Model save name",
)


args = parser.parse_args()


# Setup hyperparameters
NUM_EPOCHS: int = args.num_epochs
BATCH_SIZE: int = args.batch_size
LEARNING_RATE: float = args.learning_rate
WEIGHT_DECAY: float = args.weight_decay
LR_STEP_INTERVAL: int = args.lr_step_interval
UNFROZEN_BACKBONE_BLOCKS: List[int] = [
    int(b) for b in args.unfrozen_backbone_blocks.split(",") if b != ""
]
FROZEN_BACKBONE: bool = True if UNFROZEN_BACKBONE_BLOCKS else args.frozen_backbone
PRETRAINED_BACKBONE: bool = args.pretrained_backbone
BACKBONE_PATH: str = args.backbone_path
TARGET_TRANSFORM = args.target_transform
IMAGE_SIZE: int = args.image_size
MODEL_NAME: str = args.model_name
MODEL_SAVE_NAME: str = args.model_save_name
EXPERIMENT_NAME: str = args.experiment_name
EXPERIMENT_VARIABLE: str = args.experiment_variable


config = tools.load_config()


# Setup directories
data_path: Path = repo_root_dir / config["data_path"] / "obj_detection"
os.makedirs(data_path, exist_ok=True)

part_class_path: Path = repo_root_dir / "src" / "data" / "parts.csv"

model_save_path: Path = repo_root_dir / config["model_path"] / "obj_detection"
os.makedirs(model_save_path, exist_ok=True)
model_save_name_version: str = utils.model_save_version(
    save_dir_path=model_save_path, save_name=MODEL_SAVE_NAME
)

temp_checkpoint_path: Path = repo_root_dir / config["temp_checkpoint_path"]
os.makedirs(temp_checkpoint_path, exist_ok=True)

results_save_path: Path = repo_root_dir / config["results_path"] / "obj_detection"
os.makedirs(results_save_path, exist_ok=True)

logging_dir_path: Path = repo_root_dir / config["logging_path"] / "obj_detection"
os.makedirs(logging_dir_path, exist_ok=True)

logging_file_path: Path = logging_dir_path / (model_save_name_version + "_training.log")
os.environ["LOGGING_FILE_PATH"] = str(logging_file_path)


# Setup logging for info and debugging
logger: logging.Logger = tools.create_logger(
    log_path=logging_file_path, logger_name=__name__
)
logger.info("\n\n")
logger.info(f"Logging to file: {logging_file_path}")


# Setup SummaryWriter for tensorboards
if EXPERIMENT_NAME and EXPERIMENT_VARIABLE:
    writer: SummaryWriter | None = utils.create_writer(
        root_dir=repo_root_dir,
        experiment_name=EXPERIMENT_NAME,
        model_name=model_save_name_version,
        var=EXPERIMENT_VARIABLE,
    )
elif EXPERIMENT_NAME or EXPERIMENT_VARIABLE:
    raise NameError(
        "You need to apply a string value to both '--experiment_name' and '--experiment_variable' to use either."
    )
else:
    writer = None


# Download dataset if not already downloaded
if os.listdir(data_path):
    logger.info(
        f"There already exists files in directory: {data_path}. Assuming datasets are already downloaded!"
    )
else:
    download.kaggle_download_data(
        data_handle=config["b200_dataset_handle"],
        save_path=data_path,
        data_name=config["b200_dataset_name"],
    )


# Define paths to image and annotation data in downloaded dataset
image_dir = data_path / config["b200_dataset_name"] / "images"
annot_dir = data_path / config["b200_dataset_name"] / "annotations"


# Creating file with part id classes if not already created
class_dict: Dict[str, int] = tools.part_cat_csv_to_dict(part_class_path)


# Logging hyperparameters
logger.info(
    f"Using hyperparameters:"
    f"\n    num_epochs = {NUM_EPOCHS}"
    f"\n    batch_size = {BATCH_SIZE}"
    f"\n    learning_rate = {LEARNING_RATE}"
    f"\n    weight_decay = {WEIGHT_DECAY}"
    f"\n    lr_step_interval = {LR_STEP_INTERVAL}"
    f"\n    frozen_backbone = {FROZEN_BACKBONE}"
    f"\n    unfrozen_backbone_blocks = {UNFROZEN_BACKBONE_BLOCKS}"
    f"\n    pretrained_backbone = {PRETRAINED_BACKBONE}"
    f"\n    backbone_path = {BACKBONE_PATH}"
    f"\n    target_transform = {TARGET_TRANSFORM}"
    f"\n    image_size = {IMAGE_SIZE}"
    f"\n    model_name = {MODEL_NAME}"
    f"\n    model_save_name = {MODEL_SAVE_NAME}"
    f"\n    experiment_name = {EXPERIMENT_NAME}"
    f"\n    experiment_name = {EXPERIMENT_VARIABLE}"
)


# Setup target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device = {device}")

assert device == torch.device(
    "cuda"
), f"Device is strongly recommended to be cuda, but only available was {device}"

GPU_COUNT = torch.cuda.device_count()
logger.debug(f"gpu_count = {GPU_COUNT}")

if GPU_COUNT < 2:
    logger.warning(
        f"Amount of available GPUs is recommended to be two or more, but only got {GPU_COUNT}"
    )

dist.init_process_group("nccl")
RANK = dist.get_rank()
logger.debug(f"gpu_count = {GPU_COUNT}")


# Create a manual transform for the images if it is wanted to use that
image_transform = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, interpolation=cv2.INTER_CUBIC),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
    ),
)


def num_to_cat(target):
    return tools.get_part_cat(target, class_dict)


if TARGET_TRANSFORM:
    target_transform = lambda x: num_to_cat(x)
else:
    target_transform = None


# Create train/test dataloader
logger.debug(f"Creating dataloaders...")

train_dataloader, test_dataloader, dataset = build_features.create_dataloaders(
    image_dir=image_dir,
    annot_dir=annot_dir,
    transform=image_transform,
    target_transform=target_transform,
    batch_size=BATCH_SIZE,
)

logger.debug(f"Successfully created dataloaders.")


# Create the object detection model
logger.info("Loading model...")

objdet_model, auto_transform = models.effdet_create_model(
    model_name=MODEL_NAME,
    num_classes=len(dataset.transformed_classes),
    device=device,
    pretrained_backbone=PRETRAINED_BACKBONE,
    backbone_checkpoint_path=BACKBONE_PATH,
    frozen_backbone=FROZEN_BACKBONE,
    unfrozen_backbone_blocks=UNFROZEN_BACKBONE_BLOCKS,
    max_det_per_image=400,
)

frozen_blocks: List[str] = [
    str(i)
    for i, block in enumerate(objdet_model.backbone.blocks)
    if not all([parameter.requires_grad for parameter in block.parameters()])
]
unfrozen_blocks: List[str] = [
    str(i)
    for i, block in enumerate(objdet_model.backbone.blocks)
    if all([parameter.requires_grad for parameter in block.parameters()])
]

logger.info(
    f"Successfully loaded model: {objdet_model.__class__.__name__}"
    f"\n    Frozen blocks in backbone (not trainable): {', '.join(frozen_blocks)}"
    f"\n    Unfrozen blocks in backbone (trainable): {', '.join(unfrozen_blocks)}"
)


# Set loss, optimizer and learning rate scheduling
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    objdet_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=LR_STEP_INTERVAL,
    gamma=0.1,
)


# Train model with the training loop
logger.info("Starting training...\n")

early_stopping = utils.EarlyStopping(patience=5, delta=0.001)

# Set up scaler for better efficiency

ddp_objdet_model = DDP(objdet_model, device_ids=[device])

scaler = timm.utils.NativeScaler()


results, best_state = engine.train(
    model=ddp_objdet_model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    lr_scheduler=lr_scheduler,
    epochs=NUM_EPOCHS,
    device=device,
    temp_checkpoint_file_path=temp_checkpoint_path / (model_save_name_version + ".pt"),
    writer=writer,
    early_stopping=early_stopping,
    scaler=scaler,
)


# Save the trained model
utils.save_model(
    model=best_state,
    target_dir_path=model_save_path,
    model_name=model_save_name_version + ".pt",
)


# Save training results
results_json = json.dumps(results, indent=4)

with open(results_save_path / (model_save_name_version + "_results.json"), "w") as f:
    f.write(results_json)
