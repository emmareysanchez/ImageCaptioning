# Deep learning libraries
import torch
import numpy as np
import torchvision.transforms as transforms

# Libraries for Evaluation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider

# Libraries for Data processing
from torch.utils.data import DataLoader

# Libraries for Visualization
import matplotlib.pyplot as plt
from PIL import Image

# Other libraries
import os
import random
from typing import List, Optional
import gdown
import zipfile
from gensim.models import KeyedVectors

# Own modules
from src.data_processing import (
    download_and_prepare_dataset,
    divide_captions
)
from src.data import CollateFn, ImageAndCaptionsDataset, Vocabulary


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed (int): seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def load_data(
    path: str,
    dataset_name: str,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    This function loads the data and preprocesses it.

    Args:

        path (str): path to the data.
        dataset_name (str): name of the dataset. It can be 'flickr8k' or 'flickr30k'.
        batch_size (int): size of the batch.
        shuffle (bool): whether to shuffle the data.
        drop_last (bool): whether to drop the last batch if it is smaller
        than the batch size.
        num_workers (int): number of workers to load the data.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader, dict, dict]: tuple with
        the training, validation and test dataloaders, and the
        vocabulary of the training set.
    """
    # Download and devide images into train, val and test
    download_and_prepare_dataset(path, dataset_name)

    # Load and process captions into txt files
    captions_path = path + '/' + dataset_name
    divide_captions(captions_path)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    transform_val_test = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )

    train_path_c = f"{path}/{dataset_name}/captions_train.txt"
    val_path_c = f"{path}/{dataset_name}/captions_val.txt"
    test_path_c = f"{path}/{dataset_name}/captions_test.txt"

    train_path_i = f"{path}/{dataset_name}/train"
    val_path_i = f"{path}/{dataset_name}/val"
    test_path_i = f"{path}/{dataset_name}/test"

    # Define the datasets
    train_dataset = ImageAndCaptionsDataset(train_path_c,
                                            train_path_i,
                                            transform=transform_train)

    # Since we will only consider the words in the train set
    # as vocab we will pass it to the validation and test
    # datasets

    pad_idx = train_dataset.vocab.word2idx["<PAD>"]
    vocab = train_dataset.vocab

    val_dataset = ImageAndCaptionsDataset(val_path_c,
                                          val_path_i,
                                          transform=transform_val_test,
                                          vocab=vocab)
    test_dataset = ImageAndCaptionsDataset(test_path_c,
                                           test_path_i,
                                           transform=transform_val_test,
                                           vocab=vocab)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=CollateFn(pad_idx))

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=CollateFn(pad_idx))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=CollateFn(pad_idx))

    return train_loader, val_loader, test_loader, vocab


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    name: str = "checkpoint",
) -> None:
    """
    This function saves a checkpoint of the model and optimizer.

    Args:
        model (torch.nn.Module): model to save.
        optimizer (torch.optim.Optimizer): optimizer to save.
        epoch (int): epoch number.
        path (str): path to save the checkpoint.
    """
    # Create folder if it does not exist
    if not os.path.isdir(path):
        os.makedirs(path)

    # Save the checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"{path}/{name}.pth",
    )
    print(f"Checkpoint saved at '{path}/{name}.pth'")
    return None


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    name: str = "checkpoint",
) -> tuple[int, torch.nn.Module, Optional[torch.optim.Optimizer]]:
    """
    This function loads a checkpoint of the model and optimizer.

    Args:
        model (torch.nn.Module): model to load.
        optimizer (torch.optim.Optimizer): optimizer to load.
        path (str): path to load the checkpoint.

    Returns:
        tuple[int, torch.nn.Module, torch.optim.Optimizer]: epoch number,
        model and optimizer / None.
    """

    # Load the checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"{path}/{name}.pth", map_location=device)

    # Load the model and optimizer
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], model, optimizer


def calculate_bleu(refs: dict, hypos: dict) -> dict:
    """
    Calculate BLEU score for a single candidate caption against
    multiple reference captions.

    Args:
        refs Dict[str, List[str]]: A list of reference captions.
        hypos Dict[str, List[str]]: A list of candidate captions.

    Returns:
        dict: BLEU score for each n-gram.
    """

    bleu_dict: dict = {'1-gram': [],
                       '2-gram': [],
                       '3-gram': [],
                       '4-gram': []}

    smoothing = SmoothingFunction().method1
    # smoothing = SmoothingFunction().method7

    weights = {
            '1-gram': (1, 0, 0, 0),
            '2-gram': (0.5, 0.5, 0, 0),
            '3-gram': (0.33, 0.33, 0.33, 0),
            '4-gram': (0.25, 0.25, 0.25, 0.25)
        }

    for img_id in hypos.keys():

        # The hypos[img_id] is a list with only one element
        # that is the caption predicted by the model
        hypo_tokens = hypos[img_id][0].split()

        # The refs[img_id] is a list with possible
        # descriptions of the image
        refs_tokens = [ref.split() for ref in refs[img_id]]

        for key, value in weights.items():
            bleu = sentence_bleu(refs_tokens,
                                 hypo_tokens,
                                 weights=value,
                                 smoothing_function=smoothing)
            bleu_dict[key].append(bleu)

    bleu_means = {key: np.mean(value) for key, value in bleu_dict.items()}

    # Return the average BLEU score
    return bleu_means


def calculate_cider(refs: dict, hypo: dict) -> float:
    """
    Calculate CIDEr score for a set of hypotheses against references.

    Args:
        refs Dict[str, List[str]]: A list of reference captions.
        hypo Dict[str, List[str]]: A list of candidate captions.

    Returns:
        float: CIDEr score
    """
    # Pass the hypothesis to list
    cider = Cider()
    score, _ = cider.compute_score(refs, hypo)
    return score


def save_image(inputs: torch.Tensor,
               caption: str,
               real_caption: str,
               folder: str = "solution",
               name: str = "image",
               mean: List = [0.485, 0.456, 0.406],
               std: List = [0.229, 0.224, 0.225]):
    """
    This function saves an image with the caption predicted by the model.

    Args:
        inputs (torch.Tensor): input image.
        caption (str): predicted caption.
        real_caption (str): real caption.
        folder (str): folder to save the image.
        name (str): name of the image.
        mean (list): mean of the image.
        std (list): standard deviation of the image.
    """

    words = caption.split()

    # Add \n every 10 words
    caption = ""
    for j, word in enumerate(words):
        caption += word + " "
        if j % 10 == 0 and j != 0:
            caption += "\n"

    image = inputs.squeeze(0).cpu().numpy().transpose((1, 2, 0))

    # Denormalize the image
    image = image * np.array(std) + np.array(mean)

    image = (image * 255).astype(np.uint8)  # Assuming image was normalized
    image = Image.fromarray(image)

    plt.figure()
    plt.imshow(image)
    plt.title(f"Predicted: {caption}\nReal: {real_caption}", fontsize=8)
    plt.axis('off')
    plt.tight_layout(pad=3.0)
    plt.savefig(f"{folder}/image_{name}.png", dpi=300)
    plt.close()


def download_embeddings() -> KeyedVectors:
    """
    This function downloads the embeddings from Google Drive
    and saves them in the NLP_DATA folder.

    Returns:
        KeyedVectors: word2vec model.
    """

    # Solo descargar si no existe el archivo
    if not os.path.exists('NLP_DATA'):

        # Google Drive direct download link
        url = 'https://drive.google.com/uc?id=1zQRH1zYBHJ_vU_uMkKvvvwQiZwP5N7wW'

        # Destination file name
        output = 'NLP_DATA.zip'

        # Download the file
        gdown.download(url, output, quiet=False)

        # Unzip the downloaded file
        with zipfile.ZipFile(output, 'r') as zip_ref:
            # zip_ref.extractall('.')
            # Only extract the embeddings folder
            for file in zip_ref.namelist():
                if 'embeddings' in file:
                    zip_ref.extract(file)

        # Remove the zip file
        os.remove(output)

    print("Creating the word2vec model.")
    path = "NLP_DATA/embeddings/GoogleNews-vectors-negative300.bin.gz"
    w2v_model = KeyedVectors.load_word2vec_format(path,
                                                  binary=True)

    print("Embeddings downloaded and saved.")
    return w2v_model
