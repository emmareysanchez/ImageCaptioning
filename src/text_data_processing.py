from typing import List, Tuple, Dict
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from src.data import MSCOCODataset
import os
import pandas as pd
import json

from src.utils import tokenize

def organize_caption_flickr8k(path: str) -> None:
    """"
    Organize the captions from the Flickr8k dataset into a single file.
    """
    df_captions = pd.read_csv(os.path.join(path, 'captions.txt'), sep=',')


def organize_caption_mscoco(path: str) -> None:
    """"
    Organize the captions from the MSCOCO dataset into a single file.
    """
    df_captions = pd.read_csv(os.path.join(path, 'captions.txt'), sep=',')

    # Crear un archivo de texto con las rutas de las imágenes y sus respectivas etiquetas
    with open(os.path.join(path, 'captions.txt'), 'w') as f:
        for i in range(len(df_captions)):
            f.write(df_captions.iloc[i, 0] + '\t' + df_captions.iloc[i, 1] + '\n')
    '''"annotations": [
        {
            "image_id": 203564,
            "id": 37,
            "caption": "A bicycle replica with a clock as the front wheel."
        },
        {
            "image_id": 322141,
            "id": 49,
            "caption": "A room with blue walls and a white sink and door."
        },
       ]'''

def create_json(file_path):

    # Initialize a dictionary to hold the image numbers as keys and a list of associated captions as values
    captions_dict = {}

    # Open and read the file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)  # Skip the header line
        for line in file:
            # Split each line into image number and caption
            image_num, caption = line.strip().split(',', 1)
            # Remove the file extension from image number
            image_num = image_num.split('.')[0]
            
            # Append the caption to the list of captions for the current image number, creating a new list if necessary
            if image_num in captions_dict:
                captions_dict[image_num].append(caption.strip())
            else:
                captions_dict[image_num] = [caption.strip()]

    # Convert the dictionary to a JSON string
    json_output = json.dumps(captions_dict, indent=4)

    # Define the output JSON file path
    output_json_path = os.path.join('./data/flickr8k', 'captions.json')

    # Write the JSON string to the output file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_output)

    return output_json_path



def load_image_captions_from_json(path: str) -> dict:
    """
    Load image captions from a captions.json file.

    Args:
        path: The path to the directory containing the captions.json file.

    Returns:
        A dictionary where keys are image file names without their extensions and values are lists of captions.
        Additionally returns a list of all words used in the captions.
    """
    # El archivo JSON ya contiene los nombres de las imágenes sin la extensión .jpeg
    json_path = os.path.join(path, 'captions.json')
    word_list = []

    with open(json_path, 'r', encoding='utf-8') as json_file:
        captions_dict = json.load(json_file)

    # Procesar cada caption para generar la word_list
    for captions in captions_dict.values():
        for caption in captions:
            # Asumiendo que quieres procesar el texto de manera similar (minúsculas, tokens, etc.)
            caption_processed = caption.lower().replace('"', '').replace("'", '')
            if caption_processed[-1] == '.':
                caption_processed = caption_processed[:-1]
            # La función tokenize debe estar definida previamente o reemplazada por el procesamiento deseado
            caption_processed = tokenize(caption_processed)
            caption_processed = caption_processed.strip()
            caption_processed = " ".join(caption_processed.split())
            caption_processed = "<s> " + caption_processed + " </s>"
            word_list.extend(caption_processed.split())

    print(f"Loaded {len(captions_dict)} image captions from {json_path}")
    return captions_dict, word_list


def load_and_process_captions_flickr8k(path: str) -> Tuple[dict, List[str]]:
    """
    Load and process image captions from the Flickr8k dataset.

    Args:
        path: The path to the directory containing the captions.json file.

    Returns:
        A tuple containing a dictionary where keys are image file names and values are lists of captions,
        and a list of all words used in the captions.
    """
    organize_caption_flickr8k(path)
    json_path = create_json(os.path.join(path, 'captions.txt'))
    captions_dict, word_list = load_image_captions_from_json(json_path)
    return captions_dict, word_list


# def load_image_captions(path: str) -> dict:
#     """
#     Load image captions from a captions.txt file.

#     Args:
#         path: The path to the directory containing the captions.txt file.

#     Returns:
#         A dictionary where keys are image file names and values are captions.
#     """
#     captions_path = os.path.join(path, 'captions.txt')
#     captions_dict = {}
#     word_list = []

#     with open(captions_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             # Cada línea contiene la ruta de la imagen seguida de la etiqueta, separadas por una coma
#             image_path, caption = line.strip().split(',')

#             # Extraer solo el nombre del archivo de imagen desde la ruta completa
#             image_name = image_path.split('#')[0]  # Asumiendo que cada imagen puede tener múltiples descripciones, separadas por '#'

#             # Preprocess the caption text
#             caption = caption.lower()

#             # Remove "" and '' from the caption
#             caption = caption.replace('"', '')
#             caption = caption.replace("'", '')

#             # Remove the dot at the end of the sentence if it exists
#             if caption[-1] == '.':
#                 caption = caption[:-1]

#             # Replaces punctuation marks with respective tokens
#             caption = tokenize(caption)

#             # Delete posible spaces at the beginning and end of the caption
#             caption = caption.strip()

#             # Check there are no double spaces in the caption and remove them
#             caption = " ".join(caption.split())

#             # Add start and end tokens to the caption
#             caption = "<s> " + caption + " </s>"

#             # Agregar la etiqueta al diccionario, agrupando las etiquetas por nombre de archivo de imagen
#             if image_name in captions_dict:
#                 captions_dict[image_name].append(caption)
#             else:
#                 captions_dict[image_name] = [caption]
    
#     # agregar a la lista de palabras todas las palabras de cada caption
#     for key in captions_dict.keys():
#         for caption in captions_dict[key]:
#             word_list.extend(caption.split())

#     json_path = os.path.join(path, 'captions.json')
#     with open(json_path, 'w', encoding='utf-8') as json_file:
#         json.dump(captions_dict, json_file, ensure_ascii=False, indent=4)

#     return captions_dict, word_list




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


if __name__ == "__main__":

    # Uso de la función
    path = "./data/flickr8k"
    captions_dict = load_image_captions(path)