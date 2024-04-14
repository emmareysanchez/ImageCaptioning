from typing import List, Tuple, Dict
from collections import Counter
import os
import pandas as pd
import json
import shutil


def tokenize(text: str) -> str:
    """
    Tokenize a text by replacing punctuation with tokens.

    Args:
        text (str): text to tokenize.

    Returns:
        str: Tokenized text.
    """

    # Replace punctuation with tokens so we can use them in our model
    text = text.replace(".", " <PERIOD> ")
    text = text.replace(",", " <COMMA> ")
    text = text.replace('"', " <QUOTATION_MARK> ")
    text = text.replace(";", " <SEMICOLON> ")
    text = text.replace("!", " <EXCLAMATION_MARK> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace("(", " <LEFT_PAREN> ")
    text = text.replace(")", " <RIGHT_PAREN> ")
    text = text.replace("--", " <HYPHENS> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace(":", " <COLON> ")

    return text


def untokenize(text: str) -> str:
    """
    Untokenize a text by replacing tokens with punctuation.

    Args:
        text (str): text to untokenize.

    Returns:
        str: Untokenized text.
    """

    # Replace punctuation with tokens so we can use them in our model
    text = text.replace(" <PERIOD> ", ".")
    text = text.replace(" <COMMA> ", ",")
    text = text.replace(" <QUOTATION_MARK> ", '"')
    text = text.replace(" <SEMICOLON> ", ";")
    text = text.replace(" <EXCLAMATION_MARK> ", "!")
    text = text.replace(" <QUESTION_MARK> ", "?")
    text = text.replace(" <LEFT_PAREN> ", "(")
    text = text.replace(" <RIGHT_PAREN> ", ")")
    text = text.replace(" <HYPHENS> ", "--")
    text = text.replace(" <QUESTION_MARK> ", "?")
    text = text.replace(" <COLON> ", ":")

    return text


def organize_caption_flickr8k(path: str) -> None:
    """"
    Organize the captions from the Flickr8k dataset into JSON files for
    each set.

    Args:
        path (str): path where the data was downloaded.
    """
    # Open the captions.txt file as a DataFrame
    captions_path = path + "/captions.txt"

    # Separate the images into train, validation and test sets
    # Only if the images are not already separated
    if os.path.exists(captions_path):
        df_captions = pd.read_csv(captions_path, sep=',')

        images_train = os.listdir(path + '/train')
        images_val = os.listdir(path + '/val')
        images_test = os.listdir(path + '/test')

        # Filter the captions for each set
        df_captions_train = df_captions[df_captions['image'].isin(images_train)]
        df_captions_val = df_captions[df_captions['image'].isin(images_val)]
        df_captions_test = df_captions[df_captions['image'].isin(images_test)]

        # Create JSON for each set
        sets = {'train': df_captions_train,
                'val': df_captions_val,
                'test': df_captions_test}
        for set_name, set_captions in sets.items():
            # Group captions by image and convert to dictionary
            image_captions = (set_captions.groupby('image')['caption']
                              .apply(list).to_dict())

            # Write JSON file
            with open(os.path.join(path, f'captions_{set_name}.json'), 'w') as json_file:
                json.dump(image_captions, json_file, indent=4)

        # Remove the captions.txt file
        os.remove(os.path.join(path, 'captions.txt'))


def organize_caption_mscoco(path: str) -> None:
    """
    Organize the captions from the MSCOCO dataset into JSON files for
    each set.

    Args:
        path (str): path where the data was downloaded.
    """
    # FIXME: Modify this function to reorganize the images from the train and val sets
    # into train, val and test sets.
    path_in_dir = os.path.join(path, 'coco2017/annotations')

    sets = {'train': os.path.join(path_in_dir, 'captions_train2017.json'),
            'val': os.path.join(path_in_dir, 'captions_val2017.json')}

    for set_name, set_path in sets.items():
        with open(set_path, 'r') as json_file:
            json_data = json.load(json_file)

        image_ids_to_filenames = {image['id']: image['file_name']
                                  for image in json_data['images']}

        image_captions = {}

        for annotation in json_data['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']
            filename = image_ids_to_filenames.get(image_id)
            if filename:
                # Append caption to list of captions for the image (if the image exists
                # in the dictionary, it appends the caption to the list, otherwise it
                # creates a new list with the caption as the first element)
                image_captions.setdefault(filename, []).append(caption)

        with open(os.path.join(path, f'captions_{set_name}.json'), 'w') as json_file:
            json.dump(image_captions, json_file, indent=4)

    # Remove the annotations folder
    shutil.rmtree(path_in_dir)


def load_image_captions_from_json(path: str) -> dict:
    """
    Load image captions from a captions.json file.

    Args:
        path (str): The path to the directory containing the captions.json
        file.

    Returns:
        dict: A dictionary containing the image captions. where keys are
        image file names without their extensions and values are lists of
        captions. Additionally returns a list of all words used in the captions.
    """
    # El archivo JSON ya contiene los nombres de las imágenes sin la extensión .jpeg
    # json_path = os.path.join(path, 'captions.json')
    word_list = []

    with open(path, 'r', encoding='utf-8') as json_file:
        captions_dict = json.load(json_file)

    # Procesar cada caption para generar la word_list
    for captions in captions_dict.values():
        for caption in captions:
            # Asumiendo que quieres procesar el texto de manera similar
            # (minúsculas, tokens, etc.)
            caption_processed = caption.lower().replace('"', '').replace("'", '')
            if caption_processed[-1] == '.':
                caption_processed = caption_processed[:-1]
            # La función tokenize debe estar definida previamente o reemplazada por
            # el procesamiento deseado
            caption_processed = tokenize(caption_processed)
            caption_processed = caption_processed.strip()
            caption_processed = " ".join(caption_processed.split())
            caption_processed = "<s> " + caption_processed + " </s>"
            word_list.extend(caption_processed.split())

            # Update the caption in the dictionary
            captions[captions.index(caption)] = caption_processed

    print(f"Loaded {len(captions_dict)} image captions from {path}")
    return captions_dict, word_list


def load_and_process_captions_flickr8k(path: str) -> Tuple[dict, dict, dict, List[str]]:
    """
    Load and process image captions from the Flickr8k dataset.

    Args:
        path (str): The path to the directory containing the captions.json
        file.

    Returns:
        Tuple[dict, dict, dict, List[str]]: A tuple containing three
        dictionaries with the image captions for the train, validation and
        test sets, and a list of all words used in the captions.
    """
    organize_caption_flickr8k(path)

    # json_path = create_json(os.path.join(path, 'captions.txt'))
    json_path_train = path + "/captions_train.json"
    json_path_val = path + "/captions_val.json"
    json_path_test = path + "/captions_test.json"

    captions_dict_train, word_list = load_image_captions_from_json(json_path_train)
    captions_dict_val, _ = load_image_captions_from_json(json_path_val)
    captions_dict_test, _ = load_image_captions_from_json(json_path_test)

    return captions_dict_train, captions_dict_val, captions_dict_test, word_list


def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words (List[str]): A list of words from which to create vocabulary.

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: A tuple containing two
        dictionaries. The first dictionary maps words to integers, and
        the second dictionary maps integers to words.
    """
    word_counts: Counter = Counter(words)
    sorted_vocab: List[int] = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_vocab: Dict[int, str] = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int: Dict[str, int] = {word: ii for ii, word in int_to_vocab.items()}

    # Add the PAD token
    vocab_to_int['<PAD>'] = len(vocab_to_int)
    int_to_vocab[len(int_to_vocab)] = '<PAD>'

    return vocab_to_int, int_to_vocab


if __name__ == "__main__":

    # Uso de la función
    path = "./data/flickr8k"
    captions_dict = load_and_process_captions_flickr8k(path)
