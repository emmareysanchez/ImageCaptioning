# deep learning libraries
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# other libraries
import os
from PIL import Image


class ImageAndCaptionsDataset(Dataset):
    """
    This class is the Image and Captions Dataset.

    Attributes:
        path (str): path of the dataset.
        names (list): list with the names of the images.
        captions (dict): dictionary with the captions, where the key is the
        name of the image and the value is a list with the captions.
    """

    def __init__(self, path: str, captions_dict: dict) -> None:
        """
        Constructor of Flikr8kDataset.

        Args:
            path (str): path of the dataset.
            captions_dict (dict): dictionary with the captions, where the
            key is the name of the image and the value is a list with the
            captions.
        """
        self.path = path
        self.names = os.listdir(path)
        self.captions = captions_dict

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method loads an item based on the index.

        Args:
            index (int): index of the item to load.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple with the image and
            the captions. Image dimensions: [channels, height, width].
        """
        # Load image path and captions
        image_path: str = f"{self.path}/{self.names[index]}"
        captions: list = self.captions[self.names[index]]
        captions: torch.Tensor = torch.tensor(captions)

        # Load image transforming it to tensor
        transformations = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path)
        image = transformations(image)

        # Return image and captions
        return image, captions
