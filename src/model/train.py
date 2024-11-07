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

from src.data import download
from src.common import utils, tools
from src.preprocess import build_features
import engine, model

config = tools.load_config()

# Setup directories
data_path: Path = repo_root_dir / config["data_path"]
image_path: Path = data_path / config["data_name"]

model_save_path: Path = repo_root_dir / config["model_path"]

logging_path: Path = repo_root_dir / config["logging_path"]
os.makedirs(logging_path, exist_ok=True)


# Setup logging for info and debugging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  -  %(name)s  -  %(levelname)s:    %(message)s",
    handlers=[
        logging.FileHandler(logging_path / "training.log"),
        logging.StreamHandler(),
    ],
)


"""
Uncomment the comment bellow to download dataset when running train.py, if dataset is not already downloaded.
Dataset can also be downloaded from '/src/data/download.py'.
"""

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
        logging_dir_path=logging_path,
    )


# Setup hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_NAME = "efficientnet_b3_lego_sorter.pth"


# Setup target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"  Using device = {device}")


# Create the machine learning model
class_names: list[str] = os.listdir(image_path)

logger.info("  Loading model...")

efficientnet_b3, weights = model.get_model_efficientnet_b3(
    class_names=class_names, device=device
)

logger.info(f"  Successfully loaded model: {efficientnet_b3.__class__.__name__}")


# Create a manual transform for the images if it is wanted to use that
manual_transforms = transforms.Compose(
    [
        transforms.Resize(size=(240, 240)),
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
optimizer = torch.optim.Adam(efficientnet_b3.parameters(), lr=LEARNING_RATE)


# Train model with the training loop
logger.info("  Starting training...\n")

engine.train(
    model=efficientnet_b3,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
    logging_dir_path=logging_path,
)


# Save the trained model
utils.save_model(
    model=efficientnet_b3,
    target_dir_path=model_save_path,
    model_name=MODEL_SAVE_NAME,
    logging_dir_path=logging_path,
)
