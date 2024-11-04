import torch
from torchvision import transforms

import os
import sys
from pathlib import Path

repo_root_dir: str = os.path.dirname(os.path.abspath(__file__))

while repo_root_dir and os.path.basename(repo_root_dir) != "lego-sorter":
    repo_root_dir = os.path.dirname(repo_root_dir)

if repo_root_dir:
    sys.path.append(repo_root_dir)
else:
    raise FileNotFoundError("Could not find 'lego-sorter' directory in path hierarchy.")

from src.data import download
from src.common import utils, tools
from src.preprocess import build_features
import engine, src.model.model as model

config = tools.load_config()


"""
Uncomment the comment bellow to download dataset when running train.py, if dataset is not already downloaded.
Dataset can also be downloaded from '/src/data/download.py'.
"""
"""
dataset_path = os.path.join(current_dir, config["data_dir"], config["data_name"])

if os.path.isdir(dataset_path):
    print(f"'{dataset_path}' directory exists. Assuming dataset is already downloaded!")
else:
    data_handle = config["data_handle"]
    data_name = config["data_name"]
    save_path = os.path.join(current_dir, config["data_dir"])
    # print(f"Downloadting files from: {data_handle}, named: {data_name}, to: {save_path}")
    download.download_data(data_handle, save_path, data_name)
"""


# Setup hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_NAME = "efficientnet_b1_lego_sorter.pth"


# Setup directories
data_path = Path(repo_root_dir + config["data_dir"])
image_path: Path = data_path / "b200c-lego-classification-dataset"

class_names: list[str] = os.listdir(image_path)

# Setup target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create the machine learning model
efficientnet_b1, weights = model.get_model_efficientnet_b1(
    class_names=class_names, device=device
)


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
    data_dir=image_path, transform=auto_transforms, batch_size=BATCH_SIZE
)


# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(efficientnet_b1.parameters(), lr=LEARNING_RATE)


# Train model with the training loop
engine.train(
    model=efficientnet_b1,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
)


# Save the trained model in the directory 'lego-sorter/models'
model_save_path: str = os.path.join(repo_root_dir, config["model_path"])

utils.save_model(
    model=efficientnet_b1,
    target_dir=model_save_path,
    model_name=MODEL_SAVE_NAME,
)
