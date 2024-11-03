import yaml


def load_config():
    # Read in the configuration file
    with open("../../config.yaml") as p:
        config = yaml.safe_load(p)
    return config
