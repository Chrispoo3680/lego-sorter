# This file is meant to be used with the "rebrickable.com" API.
# If any other API is wanted to be used for categorizing the pieces, there is no guarantee the code will work without being changed.

import src.common.tools as tools
import json
import rebrick
import os
from tqdm import tqdm


def get_part_classes(data_path: str, save_path: str):

    # This is the API key from "rebrickable.com". It can be created in the profile settings under the API tab.
    # Provide it in the 'config.yaml' file.
    rebrick.init(config["rebrickable_api"])

    part_list: list[str] = os.listdir(data_path)
    part_classes = {}

    for part in tqdm(part_list):
        try:
            response = rebrick.lego.get_part(part)
            decoded = json.loads(response.read())
            part_cat_id = decoded.get(
                "part_cat_id", "Error: Part does not have a category id"
            )
            part_classes.update({part: part_cat_id})
        except Exception as e:
            print(f"\nError occured at part:  {part}\n{e}")

    json_object = json.dumps(part_classes, indent=4)

    with open(save_path, "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    config = tools.load_config()
    data_path: str = os.path.join(
        "../../", config["data_directory"], config["data_name"]
    )
    save_path: str = "part_classes.json"
    get_part_classes(data_path=data_path, save_path=save_path)
