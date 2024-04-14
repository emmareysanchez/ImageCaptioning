# deep learning libraries
import torch
import numpy as np
from torch.jit import RecursiveScriptModule
# from tkinter import Image
# from matplotlib import pyplot as plt

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
        model (torch.nn.Module): pytorch model.
        name (str): name of the model (without the extension, e.g. name.pt).
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
        name (str): name of the model to load.

    Returns:
        RecursiveScriptModule: model in torch_script format.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


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


def load_data(
    path: str,
    batch_size: int = 64,
    shuffle: bool = False,
    drop_last: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """
    This function loads the data and preprocesses it.

    Args:

        path (str): path to the data.
        batch_size (int): size of the batch.
        shuffle (bool): whether to shuffle the data.
        drop_last (bool): whether to drop the last batch if it is smaller
        than the batch size.
        num_workers (int): number of workers to load the data.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader, float, float]: tuple with
        the training, validation and test dataloaders, and the word_to_index
        and index_to_word dictionaries.
    """

    # load and preprocess data
    download_and_prepare_flickr8k_dataset(path)
    captions_path = path + '/flickr8k'
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

    max_length_test = max(
        [
            len(caption)
            for captions in captions_dict_test.values()
            for caption in captions
        ]
    )

    max_length_val = max(
        [
            len(caption)
            for captions in captions_dict_val.values()
            for caption in captions
        ]
    )

    # Update the lenght of the captions
    # Show the first element of the dictionary
    for key in captions_dict_train:
        for caption in captions_dict_train[key]:
            caption += [word_to_index["<PAD>"]] * (max_length_train - len(caption))


    for key in captions_dict_val:
        for caption in captions_dict_val[key]:
            caption += [word_to_index["<PAD>"]] * (max_length_val - len(caption))

    for key in captions_dict_test:
        for caption in captions_dict_test[key]:
            caption += [word_to_index["<PAD>"]] * (max_length_test - len(caption))

    # We only keep the first caption
    captions_dict_train = {key: captions_dict_train[key][0] for key in captions_dict_train}
    captions_dict_val = {key: captions_dict_val[key][0] for key in captions_dict_val}
    captions_dict_test = {key: captions_dict_test[key][0] for key in captions_dict_test}

    train_path = f"{path}/flickr8k/train"
    val_path = f"{path}/flickr8k/val"
    test_path = f"{path}/flickr8k/test"

    # Create for training, test and validation datasets
    train_dataset = ImageAndCaptionsDataset(train_path, captions_dict_train)
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


def captions_to_indices(captions: dict, word_to_index: dict) -> dict:
    """
    This function converts captions to indices.

    Args:
        captions (dict): dictionary with the captions.
            - the key is the image name.
            - the value is a list of captions.

        word_to_index (dict): dictionary to convert words to indices.

    Returns:
        dict: dictionary with the captions as indices.
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
        captions (dict): dictionary with the captions as indices.
        index_to_word (dict): dictionary to convert indices to words.

    Returns:
        dict: dictionary with the captions as words.
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
        path (str): path to the folder containing the images.
        model (RecursiveScriptModule): model to generate captions.
        index_to_word (dict): dictionary to convert indices to words.
        num_images (int): number of images to save.

    """
    # XXX: See if this is necessary and if not, remove it

    # # Get the list of images
    # images = os.listdir(path)

    # # Select a random sample of images
    # images = random.sample(images, num_images)

    # # Iterate over the images
    # for image in images:

    #     # Load the image
    #     image_path = path + "/" + image
    #     img = Image.open(image_path)

    #     # Generate the caption
    #     caption = model.generate_caption(img)

    #     # Save the image with the caption
    #     plt.imshow(img)
    #     plt.title(caption)
    #     plt.savefig(f"results/{image}")
    #     plt.close()

    return None


def generate_caption(output: torch.Tensor, index_to_word: dict) -> str:
    """
    This function generates a caption from the output of the model.

    Args:
        output (torch.Tensor): Tensor with the log-probabilities of the predicted
        words for each position in the sequence.
        index_to_word (dict): dictionary to convert indices to words.

    Returns:
        str: caption generated for the image.
    """

    # Get the indices of the words
    indices = torch.argmax(output, dim=1)

    # Convert indices to words until end of sentence
    caption = " "
    for index in indices[1:]:
        word = index_to_word[index.item()]
        if word == "</s>":
            break
        caption += word + " "


    return caption


def generate_caption2(self, outputs: torch.Tensor) -> List[str]:
    """
    Generate a caption for each batch of features in the input.

    Args:
        outputs (torch.Tensor): Tensor with the log-probabilities of the predicted
        words for each position in the sequence.

    Returns:
        List[str]: List of captions generated for each batch of features.
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

def generate_caption3(model, images, idx2word, word2idx, max_len=50):
    """
    Generate a caption for each image in the input.

    Args:
        model (torch.nn.Module): Model used to generate the captions.
        images (torch.Tensor): Tensor with the images to generate captions for.
        idx2word (dict): Dictionary to convert indices to words.

    Returns:
        List[str]: List of captions generated for each image.
    """
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        features = model.encoder(images.unsqueeze(0))
        states = None

        caption = torch.tensor([word2idx["<s>"]]).unsqueeze(0)

        for _ in range(max_len):
            hiddens = model.decoder(features, caption)
            predicted = hiddens.argmax(2).squeeze(0)
            predicted = predicted[-1].item()
            caption = torch.cat((caption, torch.tensor([[predicted]])), 1)
            if predicted == word2idx["</s>"]:
                break

        caption_words = [idx2word[idx] for idx in caption.squeeze(0).tolist()]
        caption_words = caption_words[1:-1]
        caption = " ".join(caption_words)
        return caption
    
def generate_caption3(model, image, idx2word, word2idx, max_len=50):
    """
    Generate a caption for a single image.

    Args:
        model (torch.nn.Module): Model used to generate the captions.
        image (torch.Tensor): Tensor with the image to generate captions for.
        idx2word (dict): Dictionary to convert indices to words.
        word2idx (dict): Dictionary to convert words to indices.
        max_len (int): Maximum length for the generated caption.

    Returns:
        str: Caption generated for the image.
    """
    model.eval()
    with torch.no_grad():
        features = model.encoder(image.unsqueeze(0))
        inputs = features
        states = None

        sampled_ids = []

        for _ in range(max_len):
            hidden, states = model.decoder.lstm(inputs, states)
            outputs = model.decoder.linear(hidden.squeeze(1))
            _, predicted = outputs.max(1)
            inputs = model.decoder.embedding(predicted)
            sampled_ids.append(predicted.item())
            if predicted.item() == word2idx["</s>"]:
                break
            print(sampled_ids)

        sampled_caption = [idx2word[idx] for idx in sampled_ids]
        return " ".join(sampled_caption)
