import yaml
from tqdm import tqdm
import logging
from pathlib import Path


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def create_logger(log_path: Path, logger_name: str):
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
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  -  %(name)s  -  %(levelname)s:    %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


def load_config():
    # Read in the configuration file
    with open("../../config.yaml") as p:
        config = yaml.safe_load(p)
    return config
