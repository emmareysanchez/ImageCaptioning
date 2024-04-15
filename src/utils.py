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
    shuffle: bool = False,
    drop_last: bool = True,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, float, float]:
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
        the training, validation and test dataloaders, and the word_to_index
        and index_to_word dictionaries.
    """
    # Download and devide images into train, val and test
    download_and_prepare_dataset(path, dataset_name)

    # Load and process captions into txt files
    captions_path = path + '/' + dataset_name
    divide_captions(captions_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
                                            transform=transform)

    # Since we will only consider the words in the train set
    # as vocab we will pass it to the validation and test
    # datasets

    pad_idx = train_dataset.vocab.word2idx["<PAD>"]
    vocab = train_dataset.vocab

    val_dataset = ImageAndCaptionsDataset(val_path_c,
                                          val_path_i,
                                          transform=transform,
                                          vocab=vocab)
    test_dataset = ImageAndCaptionsDataset(test_path_c,
                                           test_path_i,
                                           transform=transform,
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
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=CollateFn(pad_idx))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=CollateFn(pad_idx))

    # XXX: Lo mismo es mejor devolver solo el vocab
    return train_loader, val_loader, test_loader, vocab.word2idx, vocab.idx2word


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
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
        f"{path}/checkpoint.pth",
    )
    print("Checkpoint saved at 'checkpoint.pth'")
    return None


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
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
    checkpoint = torch.load(f"{path}/checkpoint.pth")

    # Load the model and optimizer
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], model, optimizer


def calculate_bleu(reference_captions: List[str], candidate_caption: str) -> float:
    """
    Calculate BLEU score for a single candidate caption against
    multiple reference captions.

    Args:
        reference_captions (List[str]): A list of reference captions.
        candidate_caption (str): The candidate caption as a string.

    Returns:
        float: BLEU score
    """
    # We tokenize the candidate caption to match the reference captions
    candidate_caption = candidate_caption.split()
    reference_captions = [ref.split() for ref in reference_captions]

    return sentence_bleu(reference_captions, candidate_caption)


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
        refs (dict): Dictionary of reference captions with image_id as keys
        and a list of captions as values.
        hypo (dict): Dictionary of hypothesis captions with image_id as keys
        and a single caption as value.

    Returns:
        float: CIDEr score
    """
    cider = Cider()
    score, scores = cider.compute_score(refs, hypo)
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


def target_caption(targets, index_to_word):
    """
    This function generates the target caption.

    Args:
        targets: The target caption.
        index_to_word: The index to word mapping.

    Returns:
        str: The target caption.
    """
    # Concat the words without the until the </s> and avoiding <s>
    caption = ""
    for i in targets[1:]:
        if index_to_word[i.item()] == "</s>":
            break
        caption += index_to_word[i.item()] + " "
    return caption
