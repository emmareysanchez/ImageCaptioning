# deep learning libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# other libraries
import os
import requests
import tarfile
import shutil
from requests.models import Response
from tarfile import TarFile
from PIL import Image


class MSCOCODataset(Dataset):
    """
    This class is the MSCOCO Dataset.
    """

    def __init__(self, path: str) -> None:
        """
        Constructor of MSCOCODataset.

        Args:
            path: path of the dataset.
        """

        # set attributes
        # TODO: Verify if this is correct or pertinent
        self.path = path
        self.names = os.listdir(path)

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """
        # TODO: Verify if this is correct or pertinent
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with image and label. Image dimensions:
                [channels, height, width].
        """
        # TODO: Verify if this is correct or pertinent
        # load image path and label
        image_path: str = f"{self.path}/{self.names[index]}"
        label: int = int(self.names[index].split("_")[0])

        # load image
        transformations = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path)
        image = transformations(image)

        return image, label


class Flikr30kDataset(Dataset):
    """
    This class is the Flikr30k Dataset.
    """

    def __init__(self, path: str) -> None:
        """
        Constructor of Flikr30kDataset.

        Args:
            path: path of the dataset.
        """

        # set attributes
        # TODO: Flikr30kDataset Verify if this is correct or pertinent
        self.path = path
        self.names = os.listdir(path)

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """
        # TODO: Flikr30kDataset Verify if this is correct or pertinent
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with image and label. Image dimensions:
                [channels, height, width].
        """
        # TODO: Flikr30kDataset Verify if this is correct or pertinent
        # load image path and label
        image_path: str = f"{self.path}/{self.names[index]}"
        label: int = int(self.names[index].split("_")[0])

        # load image
        transformations = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path)
        image = transformations(image)

        return image, label


class Flikr8kDataset(Dataset):
    """
    This class is the Flikr8k Dataset.
    """

    def __init__(self, path: str) -> None:
        """
        Constructor of Flikr8kDataset.

        Args:
            path: path of the dataset.
        """

        # set attributes
        # TODO: Flikr8kDataset Verify if this is correct or pertinent
        self.path = path
        self.names = os.listdir(path)

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """
        # TODO: Flikr8kDataset Verify if this is correct or pertinent
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with image and label. Image dimensions:
                [channels, height, width].
        """
        # TODO: Flikr8kDataset Verify if this is correct or pertinent
        # load image path and label
        image_path: str = f"{self.path}/{self.names[index]}"
        label: int = int(self.names[index].split("_")[0])

        # load image
        transformations = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path)
        image = transformations(image)

        return image, label
