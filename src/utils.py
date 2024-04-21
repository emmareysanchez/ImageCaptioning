# deep learning libraries
import torch
import numpy as np

# other libraries
import os
import random

from typing import List

# Libraries for evaluation
from nltk.translate.bleu_score import sentence_bleu  # , SmoothingFunction
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score

# Libraries for data processing
from torch.utils.data import DataLoader
from src.data_processing import (
    download_and_prepare_dataset,
    divide_captions
)
from src.data import CollateFn, ImageAndCaptionsDataset

import torchvision.transforms as transforms

# Libraries for visualization
import matplotlib.pyplot as plt
from PIL import Image

from src.data import Vocabulary

import gdown
import zipfile
from gensim.models import KeyedVectors


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
            transforms.CenterCrop((299, 299)),
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
    optimizer: torch.optim.Optimizer,
    path: str,
    name: str = "checkpoint",
) -> tuple[int, torch.nn.Module, torch.optim.Optimizer]:
    """
    This function loads a checkpoint of the model and optimizer.

    Args:
        model (torch.nn.Module): model to load.
        optimizer (torch.optim.Optimizer): optimizer to load.
        path (str): path to load the checkpoint.

    Returns:
        tuple[int, torch.nn.Module, torch.optim.Optimizer]: epoch number,
        model and optimizer.
    """
    # Load the checkpoint
    checkpoint = torch.load(f"{path}/{name}.pth", map_location="cpu")

    # Load the model and optimizer
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], model, optimizer


def calculate_bleu(refs: dict, hypos: dict) -> float:
    """
    Calculate BLEU score for a single candidate caption against
    multiple reference captions.

    Args:
        reference_captions Dict[str, List[str]]: A list of reference captions.
        hypos Dict[str, List[str]]: A list of candidate captions.

    Returns:
        float: BLEU score
    """
    bleu_scores = []
    for img_id in hypos.keys():
        bleu_for_img = 0
        for hypo in hypos[img_id]:
            bleu_score = sentence_bleu(refs[img_id], hypo)
            if bleu_score > bleu_for_img:
                bleu_for_img = bleu_score
        bleu_scores.append(bleu_for_img)
    
    return sum(bleu_scores) / len(bleu_scores)


def calculate_rouge(reference_captions: List[str], candidate_caption: str) -> float:
    """
    Calculate ROUGE score for a single candidate caption against
    multiple reference captions.

    Args:
        reference_captions (List[str]): A list of reference captions.
        candidate_caption (str): The candidate caption as a string.

    Returns:
        float: ROUGE score
    """
    rouge = Rouge()
    scores = [rouge.get_scores(candidate_caption, ref)[0] for ref in reference_captions]

    # Example to extract ROUGE-L F1 score: scores[0]['rouge-l']['f']
    # We return the average ROUGE-L F1 score across all references.

    # Hay más rouges en el diccionario y se puede cambiar el valor de 'l'
    # por '1', '2', '3'...
    return sum([score["rouge-l"]["f"] for score in scores]) / len(scores)


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


def calculate_meteor(reference_captions: List[str], candidate_caption: str) -> float:
    """
    Calculate METEOR score for a single candidate caption against
    multiple reference captions.

    Args:
        reference_captions (List[str]): A list of reference captions.
        candidate_caption (str): The candidate caption as a string.

    Returns:
        float: METEOR score
    """
    # NLTK's meteor_score function takes a list of reference captions and a candidate
    # caption, both must be strings.

    # The function calculates the METEOR score for each reference separately and returns
    # the highest score.
    scores = [meteor_score([ref], candidate_caption) for ref in reference_captions]
    # In this case, we return the average METEOR score across all references.
    return sum(scores) / len(scores)


def save_image(inputs,
               caption,
               real_caption,
               folder = "solution",
               batch_idx = 0,
               mean = [0.485, 0.456, 0.406],
               std = [0.229, 0.224, 0.225]):
    
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
    plt.savefig(f"{folder}/image_{batch_idx}.png", dpi=300)
    plt.close()


def download_embeddings():
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
            zip_ref.extractall('.')
        
    w2v_model = KeyedVectors.load_word2vec_format("NLP_Data/embeddings/GoogleNews-vectors-negative300.bin.gz",
                                                    binary=True)

    print("Embeddings downloaded and saved.")
    return w2v_model
