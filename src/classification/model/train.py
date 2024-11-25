"""
This is a file for training the lego classifier model. This file have to be run from the folder it is in.
"""

import torch
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter

import sys
from pathlib import Path

repo_root_dir: Path = Path(__file__).parent.parent.parent
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

parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--model_save_name",
    type=str,
    default="efficientnet_b0_lego_sorter",
    help="Model save name",
)
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
parser.add_argument(
    "--experiment_variable", type=str, default=None, help="Experiment variable"
)

args = parser.parse_args()


# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
MODEL_SAVE_NAME = args.model_save_name
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_VARIABLE = args.experiment_variable


config = tools.load_config()


# Setup directories
data_path: Path = repo_root_dir / config["data_path"]
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
    part_class = str(class_dict[id])
    if part_class not in class_names:
        class_names.append(part_class)
class_names.sort()


# Logging hyperparameters
logger.info(
    f"Using hyperparameters:"
    f"\n    num_epochs = {NUM_EPOCHS}"
    f"\n    batch_size = {BATCH_SIZE}"
    f"\n    learning_rate = {LEARNING_RATE}"
    f"\n    model_save_name = {MODEL_SAVE_NAME}"
    f"\n    experiment_name = {EXPERIMENT_NAME}"
    f"\n    experiment_name = {EXPERIMENT_VARIABLE}"
)


# Setup target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device = {device}")


# Create the machine learning model
logger.info("Loading model...")

cnn_model, auto_transform = model.timm_create_model(
    model_name="tf_efficientnet_b0",
    class_names=class_names,
    device=device,
    pretrained=True,
)

frozen_blocks: List[str] = [
    str(i)
    for i, block in enumerate(cnn_model.blocks)
    if not all([parameter.requires_grad for parameter in block.parameters()])
]
unfrozen_blocks: List[str] = [
    str(i)
    for i, block in enumerate(cnn_model.blocks)
    if all([parameter.requires_grad for parameter in block.parameters()])
]

logger.info(
    f"Successfully loaded model: {cnn_model.__class__.__name__}"
    f"\n    Frozen blocks in 'features' layer (not trainable): {', '.join(frozen_blocks)}"
    f"\n    Unfrozen blocks in 'features' layer (trainable): {', '.join(unfrozen_blocks)}"
)


# Create a manual transform for the images if it is wanted to use that
manual_transform = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=16),
        transforms.ToTensor(),
    ]
)

class_set = list(set(class_dict.values()))
class_dict_idx = {
    part: class_set.index(part_class) for part, part_class in class_dict.items()
}
target_transform = transforms.Lambda(lambda x: class_dict_idx[x])


# Create train/test dataloader
train_dataloader, test_dataloader = build_features.create_dataloaders(
    data_dir_path=image_paths,
    transform=auto_transform,
    target_transform=target_transform,
    batch_size=BATCH_SIZE,
)


# Set loss, optimizer and learning rate scheduling
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[11, 21, 31, 41], gamma=0.1
)


# Train model with the training loop
logger.info("Starting training...\n")
early_stopping = utils.EarlyStopping(patience=3, min_delta=0.001)

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
