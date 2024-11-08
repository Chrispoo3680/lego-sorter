"""
This is a file for training the lego classifier model. This file have to be run from the folder it is in.
"""

import torch
from torchvision import transforms


import sys
from pathlib import Path

repo_root_dir: Path = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root_dir))

import os
import logging
import argparse

from src.data import download
from src.common import utils, tools
from src.preprocess import build_features
import engine, model

config = tools.load_config()

# Setup directories
data_path: Path = repo_root_dir / config["data_path"]
image_path: Path = data_path / config["data_name"]

model_save_path: Path = repo_root_dir / config["model_path"]

logging_dir_path: Path = repo_root_dir / config["logging_path"]
os.makedirs(logging_dir_path, exist_ok=True)

logging_file_path: Path = logging_dir_path / "training.log"


# Setup logging for info and debugging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  -  %(name)s  -  %(levelname)s:    %(message)s",
    handlers=[
        logging.FileHandler(logging_file_path),
        logging.StreamHandler(),
    ],
)

logger.info("\n\n\n")


# Download dataset if not already downloaded
if os.path.isdir(image_path):
    logger.info(
        f"  '{image_path}' directory exists. Assuming dataset is already downloaded!"
    )
else:
    data_handle = config["data_handle"]
    data_name = config["data_name"]
    download.download_data(
        data_handle=data_handle,
        save_path=data_path,
        data_name=data_name,
        logging_file_path=logging_file_path,
    )

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

args = parser.parse_args()


# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
MODEL_SAVE_NAME = args.model_save_name

logger.info(
    f"Using hyperparameters:"
    f"\n        num_epochs = {NUM_EPOCHS}"
    f"\n        batch_size = {BATCH_SIZE}"
    f"\n        learning_rate = {LEARNING_RATE}"
    f"\n        model_save_name = {MODEL_SAVE_NAME}"
)


# Setup target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"  Using device = {device}")


# Create the machine learning model
class_names: list[str] = os.listdir(image_path)

logger.info("  Loading model...")

model, weights = model.get_model_efficientnet_b0(class_names=class_names, device=device)

logger.info(f"  Successfully loaded model: {model.__class__.__name__}")


# Create a manual transform for the images if it is wanted to use that
manual_transforms = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=16),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

# Get transforms from the transfered model
auto_transforms = weights.transforms()

# Create train/test dataloader
train_dataloader, test_dataloader = build_features.create_dataloaders(
    data_dir_path=image_path, transform=auto_transforms, batch_size=BATCH_SIZE
)


# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Train model with the training loop
logger.info("  Starting training...\n")

engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
    logging_file_path=logging_file_path,
)


# Save the trained model
utils.save_model(
    model=model,
    target_dir_path=model_save_path,
    model_name=MODEL_SAVE_NAME,
    logging_file_path=logging_file_path,
)
