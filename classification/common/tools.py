import yaml
from tqdm import tqdm
import logging
from pathlib import Path
import rebrick
import json
import urllib.error
import re
from zipfile import ZipFile
import os
import pandas as pd

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record) -> None:
        msg: str = self.format(record)
        tqdm.write(msg)


def create_logger(logger_name: str, log_path: Optional[Path] = None) -> logging.Logger:
    # Set up logging
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Tqdm handler for terminal output
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(
        logging.Formatter("%(asctime)s  -  %(name)s  -  %(levelname)s:    %(message)s")
    )
    logger.addHandler(tqdm_handler)

    # File handler for .log file output
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  -  %(name)s  -  %(levelname)s:    %(message)s"
            )
        )
        logger.addHandler(file_handler)

    return logger


def load_config():
    # Read in the configuration file
    with open("../../config.yaml") as p:
        config = yaml.safe_load(p)
    return config


def rename_and_unzip_file(
    zip_file_path: Union[str, Path], new_file_path: Union[str, Path]
) -> None:
    with ZipFile(zip_file_path, "r") as zipped:
        zipped.extractall(path=new_file_path)

    os.remove(zip_file_path)


def get_part_cat(part_id: str, id_to_cat: Dict[str, int]) -> int:

    logger: logging.Logger = create_logger(logger_name=__name__)

    split_config = re.compile("([0-9]+)([a-zA-Z]+)")
    possible_letters: list[str] = ["", "a", "b", "c"]

    for part_slice in part_id.split("_"):
        part_num, part_letter = (part_slice, "") if split_config.match(part_slice) is None else split_config.match(part_slice).groups()  # type: ignore
        for letter in [part_letter] + [
            let for let in possible_letters if let != part_letter
        ]:  # If part id is not found, try to alter the part id to match with the right one
            try:
                part_cat_id: int = id_to_cat[part_num + letter]
                return part_cat_id

            except KeyError as err:
                # logger.warning(
                #    f"Part with id: {part_slice} was not recognised.\n{err}\n"
                # )
                pass

    logger.error(f"Couldn't find any part categories for part with id: {part_id}\n\n")
    return 0


def part_cat_csv_to_dict(part_to_cat_path: Union[str, Path]) -> Dict[str, int]:
    part_df = pd.read_csv(part_to_cat_path, sep=",")

    part_nums = part_df["part_num"].to_numpy()
    part_cat_ids = part_df["part_cat_id"].to_numpy()

    num_to_cat: Dict[str, int] = {num: cat for num, cat in zip(part_nums, part_cat_ids)}

    return num_to_cat
