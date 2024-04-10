# deep learning libraries
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

def tokenize(text: str) -> str:

    # Replace punctuation with tokens so we can use them in our model
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace(':', ' <COLON> ')

    return text


def untokenize(text: str) -> str:
    
    # Replace punctuation with tokens so we can use them in our model
    text = text.replace(' <PERIOD> ', '.')
    text = text.replace(' <COMMA> ', ',')
    text = text.replace(' <QUOTATION_MARK> ', '"')
    text = text.replace(' <SEMICOLON> ', ';')
    text = text.replace(' <EXCLAMATION_MARK> ', '!')
    text = text.replace(' <QUESTION_MARK> ', '?')
    text = text.replace(' <LEFT_PAREN> ', '(')
    text = text.replace(' <RIGHT_PAREN> ', ')')
    text = text.replace(' <HYPHENS> ', '--')
    text = text.replace(' <QUESTION_MARK> ', '?')
    text = text.replace(' <COLON> ', ':')

    return text


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
    return sum([score['rouge-l']['f'] for score in scores]) / len(scores)


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