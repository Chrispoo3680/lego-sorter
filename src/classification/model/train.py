"""
This is a file for training the lego classifier model.
"""

import torch
from torch import nn
from torchvision.transforms import v2
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
from torch.amp.grad_scaler import GradScaler

import sys
from pathlib import Path

repo_root_dir: Path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(repo_root_dir))

import os
import logging
from tqdm import tqdm
import argparse
import json

from src.data import download, generate_lego_part_classes
from src.common import utils, tools
from src.classification.preprocess import build_features
import engine, model

from typing import Dict, List


# Setup arguments parsing for hyperparameters
parser = argparse.ArgumentParser(description="Hyperparameter configuration")

parser.add_argument(
    "--timm_model",
    type=bool,
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Should model come from timm",
)
parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
parser.add_argument(
    "--frozen_blocks",
    type=str,
    default="",
    help="Number of blocks to be frozen on format 'block1,block2,block3'",
)
parser.add_argument(
    "--pretrained",
    type=bool,
    default=True,
    help="If model should use pretrained weights",
)
parser.add_argument("--image_size", type=int, default=None, help="Image size")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="",
    help="Path to checkpoint used to initialize model weights",
)
parser.add_argument(
    "--target_transform",
    type=bool,
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Target transform is applied",
)
parser.add_argument("--model_name", type=str, required=True, help="Loaded models name")
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
parser.add_argument(
    "--experiment_variable", type=str, default=None, help="Experiment variable"
)

args, remaining = parser.parse_known_args()

parser.add_argument(
    "--model_save_name",
    type=str,
    default=args.model_name + "_lego_classifier",
    help="Model save name",
)


args = parser.parse_args()


# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
FROZEN_BLOCKS = [int(b) for b in args.frozen_blocks.split(",") if b != ""]
PRETRAINED = args.pretrained
CHECKPOINT_PATH = args.checkpoint_path
TARGET_TRANSFORM = args.target_transform
IMAGE_SIZE = args.image_size
MODEL_NAME = args.model_name
MODEL_SAVE_NAME = args.model_save_name
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_VARIABLE = args.experiment_variable


config = tools.load_config()


# Setup directories
data_path: Path = repo_root_dir / config["data_path"] / "classification"
os.makedirs(data_path, exist_ok=True)

part_class_path: Path = repo_root_dir / "src" / "data" / "parts.csv"

model_save_path: Path = repo_root_dir / config["model_path"] / "classification"
os.makedirs(model_save_path, exist_ok=True)
model_save_name_version: str = utils.model_save_version(
    save_dir_path=model_save_path, save_name=MODEL_SAVE_NAME
)

results_save_path: Path = repo_root_dir / config["results_path"] / "classification"
os.makedirs(results_save_path, exist_ok=True)

logging_dir_path: Path = repo_root_dir / config["logging_path"] / "classification"
os.makedirs(logging_dir_path, exist_ok=True)

logging_file_path: Path = logging_dir_path / (model_save_name_version + "_training.log")


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
        logging_file_path=logging_file_path,
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
    download.api_scraper_download_data(
        download_url=config["class_scraper_dataset0_download"],
        save_path=data_path,
        data_name=config["class_scraper_dataset0_name"],
        logging_file_path=logging_file_path,
    )

    download.api_scraper_download_data(
        download_url=config["class_scraper_dataset1_download"],
        save_path=data_path,
        data_name=config["class_scraper_dataset1_name"],
        logging_file_path=logging_file_path,
    )

    download.kaggle_download_data(
        data_handle=config["b200c_dataset_handle"],
        save_path=data_path,
        data_name=config["b200c_dataset_name"],
        logging_file_path=logging_file_path,
    )


# Finding all paths to image data in downloaded datasets
image_paths: List[Path] = []
for root, dirs, _ in os.walk(data_path):
    for dir_name in dirs:
        folder_path: str = os.path.join(root, dir_name)
        subfolder_contents: List[str] = os.listdir(folder_path)

        if all(
            os.path.isfile(os.path.join(folder_path, item))
            for item in subfolder_contents
        ):
            image_paths.append(Path(root))
            break


# Creating file with part id classes if not already created
part_ids: List[str] = sorted(
    set([part for img_path in image_paths for part in os.listdir(img_path)])
)
class_names: List[str] = []

class_dict: Dict[str, int] = tools.part_cat_csv_to_dict(part_class_path)
for id in part_ids:
    part_class = str(tools.get_part_cat(part_id=id, id_to_cat=class_dict))
    if part_class not in class_names:
        if TARGET_TRANSFORM:
            class_names.append(part_class)
        else:
            class_names.append(id)
class_names.sort()


# Logging hyperparameters
logger.info(
    f"Using hyperparameters:"
    f"\n    num_epochs = {NUM_EPOCHS}"
    f"\n    batch_size = {BATCH_SIZE}"
    f"\n    learning_rate = {LEARNING_RATE}"
    f"\n    weight_decay = {WEIGHT_DECAY}"
    f"\n    frozen_blocks = {FROZEN_BLOCKS}"
    f"\n    pretrained = {PRETRAINED}"
    f"\n    checkpoint_path = {CHECKPOINT_PATH}"
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


# Create the machine learning model
if args.timm_model:
    cnn_model, auto_transform = model.get_timm_model(
        model_name=MODEL_NAME,
        num_classes=len(class_names),
        device=device,
        pretrained=PRETRAINED,
        checkpoint_path=CHECKPOINT_PATH,
        frozen_blocks=FROZEN_BLOCKS,
    )
else:
    cnn_model, auto_transform = model.get_tv_efficientnet_b0(
        num_classes=len(class_names),
        device=device,
        pretrained=PRETRAINED,
        checkpoint_path=CHECKPOINT_PATH,
        frozen_blocks=FROZEN_BLOCKS,
    )


# Create a manual transform for the images if it is wanted to use that
manual_transform: Dict[str, v2.Compose] = {
    "train": v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(
                size=(256, 256),
                interpolation=v2.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(
                size=(256, 256),
                interpolation=v2.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=True,
            ),
            v2.CenterCrop(size=(224, 224)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


image_transform = auto_transform

if IMAGE_SIZE is not None:
    for transform in image_transform.transforms:  # type: ignore
        if type(transform) in (
            transforms.transforms.Resize,
            transforms.transforms.CenterCrop,
        ):
            transform.size = IMAGE_SIZE


def num_to_cat(target):
    return tools.get_part_cat(target, class_dict)


if TARGET_TRANSFORM:
    target_transform = lambda x: num_to_cat(x)
else:
    target_transform = None


# Create train/test dataloader
train_dataloader, test_dataloader = build_features.create_dataloaders(
    data_dir_path=image_paths,
    transform=image_transform,
    target_transform=TARGET_TRANSFORM,
    batch_size=BATCH_SIZE,
)


# Set loss, optimizer and learning rate scheduling
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[11, 21, 31, 41], gamma=0.1
)


# Train model with the training loop
logger.info("Starting training...\n")
early_stopping = utils.EarlyStopping(patience=3, delta=0.001)

# Set up scaler for better efficiency
scaler = GradScaler()

results = engine.train(
    model=cnn_model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    lr_scheduler=lr_scheduler,
    epochs=NUM_EPOCHS,
    device=device,
    logging_file_path=logging_file_path,
    writer=writer,
    early_stopping=early_stopping,
    scaler=scaler,
)


# Save the trained model
utils.save_model(
    model=cnn_model,
    target_dir_path=model_save_path,
    model_name=model_save_name_version + ".pt",
    logging_file_path=logging_file_path,
)


# Save training results
results_json = json.dumps(results, indent=4)

with open(results_save_path / (model_save_name_version + "_results.json"), "w") as f:
    f.write(results_json)
