from typing import List, Tuple, Dict
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from src.data import MSCOCODataset
import os

def load_image_captions(path: str) -> dict:
    """
    Load image captions from a captions.txt file.

    Args:
        path: The path to the directory containing the captions.txt file.

    Returns:
        A dictionary where keys are image file names and values are captions.
    """
    captions_path = os.path.join(path, 'captions.txt')
    captions_dict = {}

    with open(captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Cada línea contiene la ruta de la imagen seguida de la etiqueta, separadas por una coma
            image_path, caption = line.strip().split('\t')
            # Extraer solo el nombre del archivo de imagen desde la ruta completa
            image_name = image_path.split('#')[0]  # Asumiendo que cada imagen puede tener múltiples descripciones, separadas por '#'

            # Agregar la etiqueta al diccionario, agrupando las etiquetas por nombre de archivo de imagen
            if image_name in captions_dict:
                captions_dict[image_name].append(caption)
            else:
                captions_dict[image_name] = [caption]

    return captions_dict

# Uso de la función
path = "tu/ruta/al/directorio/del/dataset"
captions_dict = load_image_captions(path)


def load_and_preprocess_data(filepath: str, start_token: str = "-", end_token: str = ".") -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        filepath (str): The path to the input file containing text data.
        start_token: str. A character used as the start token for each word.
        end_token: str. A character used as the end token for each word.

    Returns:
        List[str]: A list of tokenized words from the input text.
    """
    # load de labels from the MSCOCO dataset
    dataset: MSCOCODataset = MSCOCODataset(filepath)
    lines: List[str] = dataset

    tokens: List[str] = []
    # Preprocess and tokenize the text
    for line in lines:
        # Splitting the line by spaces and discarding the last two elements
        parts: List[str] = line.split()
        word: str = " ".join(
            parts[:-2]
        ).lower()  # Joining all parts except the last two
        # Adding start and end tokens to the word
        tokens += list(start_token) + list(word) + list(end_token)

    return tokens

def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words (List[str]): A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    word_counts: Counter = Counter(words)
    sorted_vocab: List[int] = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_vocab: Dict[int, str] = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int: Dict[str, int] = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

