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

    # Make sure the kaggle.json file is set up and permissions are correct
    kaggle.api.authenticate()

    # Create path if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Download dataset
    kaggle.api.dataset_download_files(dataset_identifier, path=path, unzip=True)
    
    # Assuming the Flickr8k dataset structure
    dataset_path = f"{path}/flickr8k"

    # Prepare directories for processed data
    if not os.path.exists(f"{path}/train"):
        os.makedirs(f"{path}/train")
    if not os.path.exists(f"{path}/val"):
        os.makedirs(f"{path}/val")

    # Define resize transformation
    transform = transforms.Resize((224, 224))

    # Process and save images
    list_splits = ("train", "val")
    for split in list_splits:
        # Adjust according to the actual Flickr8k structure on disk
        images_path = f"{dataset_path}/Images/{split}"
        list_class_dirs = os.listdir(images_path)
        
        for class_dir in list_class_dirs:
            list_images = os.listdir(f"{images_path}/{class_dir}")
            for image_file in list_images:
                image_path = f"{images_path}/{class_dir}/{image_file}"
                image = Image.open(image_path).convert('RGB')
                image = transform(image)
                image.save(f"{path}/{split}/{class_dir}_{image_file}")

    # Optional: clean up original downloaded files if desired
    # Be cautious with this, you might want to keep the original dataset
    # shutil.rmtree(dataset_path)

    print("Dataset processed and saved.")

def download_and_prepare_mscoco_dataset(
        path: str,
        start_token: str = "-") -> None:
        

    # Use the function
    path = "path/to/your/dataset/directory"
    download_and_prepare_flickr8k_dataset(path)



    def download_data(path: str) -> None:
        """
        This function downloads the data from internet.

        Args:
            path: path to dave the data.
        """

        # define paths
        url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        target_path: str = f"{path}/imagenette2.tgz"

        # download tar file
        response: Response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.raw.read())

        # extract tar file
        tar_file: TarFile = tarfile.open(target_path)
        tar_file.extractall(path)
        tar_file.close()

        # create final save directories
        os.makedirs(f"{path}/train")
        os.makedirs(f"{path}/val")

        # define resize transformation
        transform = transforms.Resize((224, 224))

        # loop for saving processed data
        list_splits: tuple[str, str] = ("train", "val")
        for i in range(len(list_splits)):
            list_class_dirs = os.listdir(f"{path}/imagenette2/{list_splits[i]}")
            for j in range(len(list_class_dirs)):
                list_dirs = os.listdir(
                    f"{path}/imagenette2/{list_splits[i]}/{list_class_dirs[j]}"
                )
                for k in range(len(list_dirs)):
                    image = Image.open(
                        f"{path}/imagenette2/{list_splits[i]}/"
                        f"{list_class_dirs[j]}/{list_dirs[k]}"
                    )
                    image = transform(image)
                    if image.im.bands == 3:
                        image.save(f"{path}/{list_splits[i]}/{j}_{k}.jpg")

        # delete other files
        os.remove(target_path)
        shutil.rmtree(f"{path}/imagenette2")

        return None

