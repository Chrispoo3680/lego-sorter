# This file is meant to be used with the "rebrickable.com" API.
# If any other API is wanted to be used for categorizing the pieces, there is no guarantee the code will work without being changed.

import sys
from pathlib import Path

repo_root_dir: Path = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root_dir))

import classification.common.tools as tools
import json
import rebrick
import os
from tqdm import tqdm
import urllib.error


def get_part_classes(save_path: Path):

    config = tools.load_config()

    # This is the API key from "rebrickable.com". It can be created in the profile settings under the API tab.
    # Provide it in the 'config.yaml' file.
    rebrick.init("8b54992fcba5941f645017f463587c76")
    # rebrick.init(config["rebrickable_api"])

    # Generate json file with all parts and respective categories in rebrickable database
    response = rebrick.lego.get_categories()
    cats = json.loads(response.read())

    part_info_dict = {}

    for cat in tqdm(cats["results"], position=0, leave=True):
        response = rebrick.lego.get_parts(part_cat_id=cat["id"], page_size=10_000)
        decoded = json.loads(response.read())
        for part in decoded["results"]:
            part_info_dict.update({part["part_num"]: part["part_cat_id"]})

    json_dict = json.dumps(part_info_dict, indent=4)

    with open(save_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    config = tools.load_config()
    save_path = Path("part_id_to_cat.json")

    get_part_classes(save_path=save_path)
