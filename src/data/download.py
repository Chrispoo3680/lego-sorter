# This file is meant to be used for downloading datasets from "kaggle.com" with the kaggle API.
# If any other datasets with potentially other API's is wanted to be used, there is no guarantee the code will work without being changed.

import sys
from pathlib import Path

repo_root_dir: Path = Path(__file__).parent.parent
sys.path.append(str(repo_root_dir))

from common import tools
import os
import logging

# To use the kaggle API you have to provide your username and a generated API key in the "kaggle_username" and "kaggle_api" variables in 'config.yaml'.
# You can get these by downloading the 'kaggle.json' file from your kaggle account by clicking on "create new token" under the API header in the settings tab.
# Your kaggle username and API key will be in the 'kaggle.json' file.


def download_data(
    data_handle: str,
    save_path: Path,
    data_name: str,
):

    config = tools.load_config()

    # Set environment variables for Kaggle authentication
    os.environ["KAGGLE_USERNAME"] = config["kaggle_username"]
    os.environ["KAGGLE_KEY"] = config["kaggle_api"]

    import kaggle

    # Download the lego piece dataset from kaggle.com
    kaggle.api.authenticate()

    logging.info(
        f"Downloading files from:  {data_handle} \nNamed:  {data_name} \nTo path:  {save_path}"
    )

    # Create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    kaggle.api.dataset_download_files(data_handle, path=save_path, unzip=True)
    os.rename(save_path / "64", save_path / data_name)

    logging.info("Successfully downloaded dataset files!")


if __name__ == "__main__":
    config = tools.load_config()
    data_handle: str = config["data_handle"]
    save_path: Path = repo_root_dir / config["data_path"]
    data_name: str = config["data_name"]
    download_data(data_handle=data_handle, save_path=save_path, data_name=data_name)
