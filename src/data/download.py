# This file is meant to be used for downloading datasets from "kaggle.com" with the kaggle API.
# If any other datasets with potentially other API's is wanted to be used, there is no guarantee the code will work without being changed.

import sys
from pathlib import Path

repo_root_dir: Path = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root_dir))

from src.common import tools
import os
import logging
import wget
import requests
from zipfile import ZipFile
from tqdm import tqdm
import io
import contextlib


# To use the kaggle API you have to provide your username and a generated API key in the "kaggle_username" and "kaggle_api" variables in 'config.yaml'.
# You can get these by downloading the 'kaggle.json' file from your kaggle account by clicking on "create new token" under the API header in the settings tab.
# Your kaggle username and API key will be in the 'kaggle.json' file.


def kaggle_download_data(
    data_handle: str,
    save_dir: Path,
    data_name: str,
    logging_file_path: Path,
    force: bool = False,
):

    config = tools.load_config()

    logger = tools.create_logger(log_path=logging_file_path, logger_name=__name__)

    buf = io.StringIO()

    # Set environment variables for Kaggle authentication
    os.environ["KAGGLE_USERNAME"] = config["kaggle_username"]
    os.environ["KAGGLE_KEY"] = config["kaggle_api"]

    import kaggle

    # Download the lego piece dataset from kaggle.com
    kaggle.api.authenticate()

    # Create path if it doesn't exist
    output_path = os.path.join(save_dir, data_name)
    os.makedirs(output_path, exist_ok=True)

    logger.info(
        f"Downloading files..."
        f"\n    From:  {data_handle}"
        f"\n    Named:  {data_name}"
        f"\n    To path:  {save_dir}"
    )

    # Get list of files in the dataset
    file_list = kaggle.api.dataset_list_files(data_handle).files

    logger.info(f"Found {len(file_list)} files in dataset '{data_handle}'")

    for f in tqdm(file_list, desc="Downloading files", leave=False):
        file_path = os.path.join(output_path, f.name)

        if os.path.exists(file_path) and not force:
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            kaggle.api.dataset_download_file(
                data_handle, f.name, path=output_path, force=force, quiet=True
            )

        # unzip if needed (since each file is zipped individually)
        zip_file = file_path + ".zip"
        if os.path.exists(zip_file):

            with ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(output_path)
            os.remove(zip_file)

    logger.info(f"Successfully downloaded dataset files from '{data_handle}'!")


def old_kaggle_download_data(
    data_handle: str,
    save_path: Path,
    data_name: str,
    logging_file_path: Path,
):

    config = tools.load_config()

    logger = tools.create_logger(log_path=logging_file_path, logger_name=__name__)

    # Set environment variables for Kaggle authentication
    os.environ["KAGGLE_USERNAME"] = config["kaggle_username"]
    os.environ["KAGGLE_KEY"] = config["kaggle_api"]

    import kaggle

    # Download the lego piece dataset from kaggle.com
    kaggle.api.authenticate()

    logger.info(
        f"Downloading files..."
        f"\n    From:  {data_handle}"
        f"\n    Named:  {data_name}"
        f"\n    To path:  {save_path}"
    )

    # Create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    kaggle.api.dataset_download_files(
        data_handle, path=save_path, unzip=False, quiet=False
    )

    zip_paths = save_path.glob("*.zip")
    zip_names = [f.name for f in zip_paths]

    if len(zip_names) == 1:
        logger.info("Zip file downloaded. Unzipping files...")
        tools.rename_and_unzip_file((save_path / zip_names[0]), (save_path / data_name))
        logger.info(f"Successfully downloaded dataset files from {data_handle}!")
    else:
        logger.error(
            f"Encountered an invalid amount of .zip files to unzip in directory: {save_path}, number on .zip files to unzip should be 1."
        )


def api_scraper_download_data(
    download_url: str,
    save_path: Path,
    data_name: str,
    logging_file_path: Path,
):

    logger: logging.Logger = tools.create_logger(
        log_path=logging_file_path, logger_name=__name__
    )

    logger.info(
        f"Downloading files..."
        f"\n    From:  {download_url}"
        f"\n    Named:  {data_name}"
        f"\n    To path:  {save_path}"
    )

    file_name: str = data_name + ".zip"

    # Create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    download_dir = wget.download(download_url, out=str(save_path))
    os.rename(save_path / download_dir, save_path / file_name)

    tools.rename_and_unzip_file(
        zip_file_path=(save_path / file_name), new_file_path=(save_path / data_name)
    )

    logger.info(f"Successfully downloaded dataset files from: {download_url}!")


if __name__ == "__main__":
    config = tools.load_config()
    save_path: Path = repo_root_dir / config["data_path"] / "testing"
    log_path = Path("download.log")

    kaggle_download_data(
        data_handle="zalando-research/fashionmnist",
        save_dir=save_path,
        data_name="fashionmnist",
        logging_file_path=log_path,
    )
