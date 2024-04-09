# deep learning libraries
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
import requests
import tarfile
import shutil
from requests.models import Response
from tarfile import TarFile
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kaggle

import os
import requests
import tarfile
import kaggle
import shutil
from PIL import Image
from torchvision import transforms


def download_and_prepare_flickr8k_dataset(path: str) -> None:
    """
    Download and prepare the Flickr8k dataset from Kaggle and process it.

    Args:
        path: Path to save the processed data.
    """

    # Kaggle dataset identifier
    dataset_identifier: str = "adityajn105/flickr8k"
    print("antes")
    # Make sure the kaggle.json file is set up and permissions are correct
    kaggle.api.authenticate()
    print("despues")

    # Create path if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    dataset_path = f"{path}/flickr8k"
    # Download dataset
    kaggle.api.dataset_download_files(dataset_identifier, path=dataset_path, unzip=True)

    print("hola")

    # Prepare directories for processed data
    if not os.path.exists(f"{dataset_path}/train"):
        os.makedirs(f"{dataset_path}/train")
    if not os.path.exists(f"{dataset_path}/val"):
        os.makedirs(f"{dataset_path}/val")

    # Define resize transformation
    transform = transforms.Resize((224, 224))

    images_list = os.listdir(f"{dataset_path}/Images")
    # Split into train and validation
    # 80% train, 20% validation
    test_images = images_list[int(len(images_list) * 0.8) :]
    train_images = images_list[: int(len(images_list) * 0.8)]
    # Of the train images, 80% will be used for training and 20% for validation
    val_images = train_images[int(len(train_images) * 0.8) :]
    train_images = train_images[: int(len(train_images) * 0.8)]
    # Process and save images
    list_splits = ["train", "val", "test"]
    list_class_dirs = [train_images, val_images, test_images]
    for i in range(len(list_splits)):
        split = list_splits[i]
        list_images = list_class_dirs[i]
        # Adjust according to the actual Flickr8k structure on disk
        images_path = f"{dataset_path}/Images"

        for image_file in list_images:
            image_path = f"{images_path}/{image_file}"
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
            image.save(f"{dataset_path}/{split}/{image_file}")

    shutil.rmtree(f"{dataset_path}/Images")

    print("Dataset processed and saved.")


if __name__ == "__main__":
    download_and_prepare_flickr8k_dataset("data")
