import os
import requests
from pathlib import Path
from PIL import Image
import zipfile

def get_data(data_path: str, image_url: str):
    """Download and unzip data from a URL."""
    data_path = Path(data_path)
    image_path = data_path/"zero2master_pytorch/going_modelar/data/pizza_steak_sushi"

    if image_path.is_dir():
        print(f"Directory '{image_path}' already exists.")
    else:
        print(f"Directory '{image_path}' does not exist, create one ...")
        image_path.mkdir(parents=True, exist_ok=True)

    # Download the data

    # image_url = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/extras/pizza_steak_sushi.zip"
    # image_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    with open(image_path/"pizza_steak_sushi.zip", "wb") as f:
        request = requests.get(image_url)
        f.write(request.content)

    with zipfile.ZipFile(image_path/"pizza_steak_sushi.zip", "r") as zip_ref:
        print(f"Uzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)

    # remove the zip file
    os.remove(image_path/"pizza_steak_sushi.zip")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Project path")
    parser.add_argument("--image_url", type=str, help="URL to download image from")
    args = parser.parse_args()
    get_data(args.data_path, args.image_url)


if __name__ == "__main__":
    main()