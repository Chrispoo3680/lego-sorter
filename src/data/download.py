""" THIS FILE MUST BE RUN FROM THE ROOT DIRECTORY (lego-sorter) OR INDIRECTLY FROM RUNNING '/src/model/train.py'"""

# This file is meant to be used for downloading datasets from "kaggle.com" with the kaggle API.
# If any other datasets with potentially other API's is wanted to be used, there is no guarantee the code will work without being changed.

from src.common import tools
import os
import kaggle


# To use the kaggle API you have to provide your username and a generated API key in the "kaggle_username" and "kaggle_api" variables in 'config.yaml'.
# You can get these by downloading the 'kaggle.json' file from your kaggle account by clicking on "create new token" under the API header in the settings tab.
# Your kaggle username and API key will be in the 'kaggle.json' file.
def download_data(
    data_handle: str,
    save_path: str,
    data_name: str,
):

    config = tools.load_config()

    # Set environment variables for Kaggle authentication
    os.environ["KAGGLE_USERNAME"] = config["kaggle_username"]
    os.environ["KAGGLE_KEY"] = config["kaggle_api"]

    # Download the lego piece dataset from kaggle.com
    kaggle.api.authenticate()

    # Create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    kaggle.api.dataset_download_files(data_handle, path=save_path, unzip=True)
    os.rename(os.path.join(save_path, "64"), os.path.join(save_path, data_name))


if __name__ == "__main__":
    config = tools.load_config()
    data_handle = config["data_handle"]
    save_path = config["data_dir"]
    data_name = config["data_name"]
    download_data(data_handle=data_handle, save_path=save_path, data_name=data_name)
