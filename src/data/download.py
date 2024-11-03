""" !THIS FILE MUST BE RUN FROM THE ROOT DIRECTORY (lego-sorter)!"""

# This file is meant to be used for downloading datasets from "kaggle.com" with the kaggle API.
# If any other datasets with potentially other API's is wanted to be used, there is no guarantee the code will work without being changed.

from src.common import tools
import os
import kaggle


# Download the "kaggle.json" file from your kaggle account by clicking on "create new token" under the API header in the settings tab.
# Place the "kaggle.json" file in the same directory as the "download.py".
def download_data(data_handle: str, save_path: str, data_name: str):
    # Download the lego piece dataset from kaggle.com
    kaggle.api.authenticate()

    # Create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    kaggle.api.dataset_download_files(data_handle, path=save_path, unzip=True)
    os.rename(os.path.join(save_path, "64"), os.path.join(save_path, data_name))


if __name__ == "__main__":
    config = tools.load_config()
    data_handle = config["datahandle"]
    save_path = config["datadirectory"]
    data_name = config["dataname"]
    download_data(data_handle, save_path, data_name)
