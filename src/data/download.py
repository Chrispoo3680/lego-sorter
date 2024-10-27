# This file is meant to be used for downloading datasets from "kaggle.com" with the kaggle API.
# If any other datasets with potentially other API's is wanted to be used, there is no guarantee the code will work without being changed.

import src.common.tools as tools
import kaggle
import os


# Download the "kaggle.json" file from your kaggle account by clicking on "create new token" under the API header in the settings tab.
# Place the "kaggle.json" file in the same directory as the "download.py".
def download_data(data_handle, save_path):
    # Download the lego piece dataset from kaggle.com
    kaggle.api.authenticate()

    # Create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    kaggle.api.dataset_download_files(data_handle, path=save_path, unzip=True)


if __name__ == "__main__":
    config = tools.load_config()
    data_handle = config["datahandle"]
    save_path = config["datadirectory"] + config["dataname"] + "/"
    download_data(data_handle, save_path)
