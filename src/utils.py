# deep learning libraries
from tkinter import Image
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random

from collections import Counter
from typing import List, Dict

# Libraries for evaluation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score

# Libraries for data processing
from torch.utils.data import DataLoader
from src.image_data_processing import download_and_prepare_flickr8k_dataset
from src.text_data_processing import (
    load_and_process_captions_flickr8k,
    create_lookup_tables,
)
from src.data import ImageAndCaptionsDataset


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
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


def calculate_bleu(reference_captions: List[str], candidate_caption: str) -> float:
    """
    Calculate BLEU score for a single candidate caption against multiple reference captions.

    Args:
    - reference_captions: A list of lists of reference captions
    - candidate_caption: A string of the candidate caption
    Returns:
    - BLEU score
    """
    # We tokenize the candidate caption to match the reference captions
    candidate_caption = candidate_caption.split()
    reference_captions = [ref.split() for ref in reference_captions]

    return sentence_bleu(reference_captions, candidate_caption)


def calculate_rouge(reference_captions: List[str], candidate_caption: str) -> float:
    """
    Calculate ROUGE score for a single candidate caption against multiple reference captions.

    Args:
    - reference_captions: A list of reference captions
    - candidate_caption: The candidate caption as a string

    Returns:
    - ROUGE score dictionary
    """
    rouge = Rouge()
    scores = [rouge.get_scores(candidate_caption, ref)[0] for ref in reference_captions]

    # Example to extract ROUGE-L F1 score: scores[0]['rouge-l']['f']
    # We return the average ROUGE-L F1 score across all references.

    # Hay más rouges en el diccionario y se puede cambiar el valor de 'l' por '1', '2', '3'...
    return sum([score["rouge-l"]["f"] for score in scores]) / len(scores)


def calculate_cider(refs: Dict, hypo: Dict) -> float:
    """
    Calculate CIDEr score for a set of hypotheses against references.

    Args:
    - refs: Dictionary of reference captions with image_id as keys and a list of captions as values.
    - hypo: Dictionary of hypothesis captions with image_id as keys and a single caption as value.

    Returns:
    - CIDEr score
    """
    cider = Cider()
    score, scores = cider.compute_score(refs, hypo)
    return score


def calculate_meteor(reference_captions: List[str], candidate_caption: str) -> float:
    """
    Calculate METEOR score for a single candidate caption against multiple reference captions.

    Args:
    - reference_captions: A list of reference captions (each as a single string).
    - candidate_caption: The candidate caption as a single string.

    Returns:
    - METEOR score as a float.
    """
    # NLTK's meteor_score function takes a list of reference captions and a candidate caption,
    # both must be strings.
    # The function calculates the METEOR score for each reference separately and returns the highest score.
    scores = [meteor_score([ref], candidate_caption) for ref in reference_captions]
    # In this case, we return the average METEOR score across all references.
    return sum(scores) / len(scores)


def load_data(
    path: str,
    batch_size: int = 64,
    shuffle: bool = False,
    drop_last: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, float, float]:

    # load and preprocess data
    download_and_prepare_flickr8k_dataset(path)
    captions_path = path
    captions_dict_train, captions_dict_val, captions_dict_test, word_list = (
        load_and_process_captions_flickr8k(captions_path)
    )


    # Create lookup tables
    word_to_index, index_to_word = create_lookup_tables(word_list)

    # Change captions to indices
    captions_dict_train = captions_to_indices(captions_dict_train, word_to_index)
    captions_dict_val = captions_to_indices(captions_dict_val, word_to_index)
    captions_dict_test = captions_to_indices(captions_dict_test, word_to_index)

    # All captions must be the same length
    # We will use the length of the longest caption
    max_length_train = max(
        [
            len(caption)
            for captions in captions_dict_train.values()
            for caption in captions
        ]
    )

    # Update the lenght of the captions
    for key in captions_dict_train:
        for caption in captions_dict_train[key]:
            caption += [word_to_index["</s>"]] * (max_length_train - len(caption))

    # We only keep the first caption
    captions_dict_train = {
        key: captions[0] for captions in captions_dict_train[key]
        for key in captions_dict_train
    }

    # FIXME: hacerlo en val y test

    train_path = f"{path}/flickr8k/train"
    val_path = f"{path}/flickr8k/val"
    test_path = f"{path}/flickr8k/test"

    # Create for training, test and validation datasets
    train_dataset = ImageAndCaptionsDataset(
        train_path, captions_dict_train
    )
    val_dataset = ImageAndCaptionsDataset(val_path, captions_dict_val)
    test_dataset = ImageAndCaptionsDataset(test_path, captions_dict_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, word_to_index, index_to_word


def captions_to_indices(captions: dict, word_to_index:dict) -> dict:
    """
    This function converts captions to indices.

    Args:
        captions: dictionary with the captions.
            - the key is the image name.
            - the value is a list of captions.
        word_to_index: dictionary to convert words to indices.

    Returns:
        dictionary with the captions as indices.
    """
    # Change captions to indices
    # The key will be the image name and the value will be a list of lists
    # with the indices of the words
    # If the word is not in the dictionary, we will ignore it
    captions_indices = {
        key: [
            [word_to_index[word] for word in caption.split() if word in word_to_index]
            for caption in captions[key]
        ]
        for key in captions
    }

    return captions_indices


def indices_to_captions(captions: dict, index_to_word: dict) -> dict:
    """
    This function converts captions from indices to words.

    Args:
        captions: dictionary with the captions as indices.
        index_to_word: dictionary to convert indices to words.

    Returns:
        dictionary with the captions as words.
    """
    # Change captions to words
    captions_words = {
        key: [
            " ".join([index_to_word[index] for index in caption])
            for caption in captions[key]
        ]
        for key in captions
    }
    return captions_words


def save_images_with_captions(
    path: str, model: RecursiveScriptModule, index_to_word: dict, num_images: int = 5
) -> None:
    """
    This function saves images with their captions generated by the model.

    Args:
        path: path to the folder containing the images.
        model: model to generate captions.
        index_to_word: dictionary to convert indices to words.
        num_images: number of images to save.
    """

    # Get the list of images
    images = os.listdir(path)

    # Select a random sample of images
    images = random.sample(images, num_images)

    # Iterate over the images
    for image in images:

        # Load the image
        image_path = os.path.join(path, image)
        img = Image.open(image_path)

        # TODO: Implement the function generate_caption before
        # # Generate the caption
        # # caption = model.generate_caption(img)

        # # Save the image with the caption
        # plt.imshow(img)
        # plt.title(caption)
        # plt.savefig(f"results/{image}")
        # plt.close()

    return None


def generate_caption(output: torch.Tensor, index_to_word: dict) -> str:
    """
    This function generates a caption from the output of the model.

    Args:
        output: output of the model.
        index_to_word: dictionary to convert indices to words.

    Returns:
        caption as a string.
    """

    # Get the indices of the words
    indices = torch.argmax(output, dim=1)

    # Convert indices to words
    caption = " ".join([index_to_word[index.item()] for index in indices])

    return caption

def generate_caption2(self, outputs):
        """
        Generate a caption for each batch of features in the input.

        Args:
            outputs (torch.Tensor): Tensor with the log-probabilities of the predicted words for each position in the sequence.

        Returns:
            List[str]: List of captions generated for each image in the batch.
        """
        # Initialize the list to store the generated captions
        captions = []
        
        # Get the predicted words for each position in the sequence
        predicted_words = outputs.argmax(2)
        
        # Iterate over the batch
        for prediction in predicted_words:
            # Initialize the caption for the current image
            caption = []
            for word_index in prediction:
                # Get the corresponding word from the vocabulary
                word = self.vocab.itos[word_index.item()]
                # Append the word to the caption
                caption.append(word)
                # If the word is the end token, stop the caption
                if word == "</s>":
                    break
            # Join the words in the caption and append it to the list
            captions.append(" ".join(caption))
        
        return captions

