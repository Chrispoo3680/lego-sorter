# This file is meant to be used with the "rebrickable.com" API.
# If any other API is wanted to be used for categorizing the pieces, there is no guarantee the code will work without being changed.

import sys
from pathlib import Path

repo_root_dir: Path = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root_dir))

import src.common.tools as tools
import json
import rebrick
import os
from tqdm import tqdm
import urllib.error


def part_class(part: str):
    for part_slice in part.split("_"):
        for fix in [
            "",
            "a",
            "b",
            "c",
        ]:  # If part id is not found, try to alter the part id to match with the right one
            try:
                response = rebrick.lego.get_part(part_slice + fix)
                decoded = json.loads(response.read())
                part_cat_id: str = decoded.get(
                    "part_cat_id", "Error: Part does not have a category id"
                )
                return part_cat_id

            except urllib.error.HTTPError as errh:
                if errh.code == 404:
                    tqdm.write(f"\nPart with id: {part} was not recognised.\n{errh}")
                elif errh.code == 400:
                    tqdm.write(
                        f"\nUser not authenticated! Please provide a valid API key.\n{errh}"
                    )
                else:
                    tqdm.write(f"\nError occured.\n{errh}")
    return None


def get_part_classes(data_path: Path, save_path: Path):

    # This is the API key from "rebrickable.com". It can be created in the profile settings under the API tab.
    # Provide it in the 'config.yaml' file.
    rebrick.init(config["rebrickable_api"])

    # Get all paths to image folders
    image_paths: list[Path] = []
    for root, dirs, _ in os.walk(data_path):
        for dir_name in dirs:
            folder_path: str = os.path.join(root, dir_name)
            subfolder_contents: list[str] = os.listdir(folder_path)

            if all(
                os.path.isfile(os.path.join(folder_path, item))
                for item in subfolder_contents
            ):
                image_paths.append(Path(root))
                break

    part_ids = set([part for img_path in image_paths for part in os.listdir(img_path)])
    part_classes: dict[str, str] = {}

    for part in tqdm(part_ids):
        part_cat_id: str | None = part_class(part)
        if part_cat_id:
            part_classes.update({part: part_cat_id})
        else:
            tqdm.write(
                f"Not able to find any parts that match or are similar to part with part id: {part}"
            )

    json_object: str = json.dumps(part_classes, indent=4)

    with open(save_path, "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    config = tools.load_config()
    data_path: Path = repo_root_dir / config["data_path"]
    save_path = Path("part_classes.json")

    get_part_classes(data_path=data_path, save_path=save_path)
