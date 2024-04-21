# deep learning libraries
import matplotlib.pyplot as plt
import torch

# other libraries
from typing import Final

# own modules
from src.utils import set_seed, load_data, load_checkpoint, save_image, calculate_bleu, calculate_cider
from src.model import ImageCaptioningModel

# TODO: Import necessary libraries
from PIL import Image
import numpy as np
import os
from collections import defaultdict

from tqdm import tqdm
import json

# static variables
DATA_PATH: Final[str] = "data"

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

def main() -> None:
    """
    This function is the main program. It loads the data, the model,
    and evaluates it.
    """

    # TODO: Make the evaluation of the model
    # Define hyperparameters
    dataset_name = "flickr30k"  # "flickr8k" or "flickr30k"
    batch_size = 1
    embedding_size = 256
    hidden_size = 256
    num_layers = 1
    GENERATE_CAPTIONS = True

    # load data
    (_, _, test_loader, vocab) = load_data(
        DATA_PATH, dataset_name, batch_size
    )

    # model = MyModel(encoder_params, decoder_params)
    model = ImageCaptioningModel(
        embedding_size,
        hidden_size,
        len(vocab),
        num_layers
    )

    _, model, _ = load_checkpoint(model, None, "checkpoint")

    captions_dir = "captions"

    # If the captions are not generated, generate them
    if not os.path.exists(captions_dir):
        GENERATE_CAPTIONS = True
        os.makedirs(captions_dir)

    if GENERATE_CAPTIONS:

        solution_dir = "solution"
        if not os.path.exists(solution_dir):
            os.makedirs(solution_dir)

        model = model.to(device)
        model.eval()

        # evaluate model
        with torch.no_grad():

            batch_idx = 0

            refs = defaultdict(list)
            hypos = defaultdict(list)

            for img_name, inputs, targets in tqdm(test_loader):

                inputs = inputs.to(device)
                targets = targets.to(device)

                # Inputs must be float
                inputs = inputs.float()
                targets = targets.long()

                targets = targets.squeeze(1)
                real_caption = vocab.indices_to_caption(targets.tolist())

                img_id = img_name[0]
                refs[img_id].append(real_caption)

                if batch_idx % 5 == 0:

                    # Only generate the caption ones for the five images
                    # that are the same
                    caption = model.generate_caption(inputs, vocab)

                    # Add the caption to the hypos
                    hypos[img_id].append(caption)

                    words = caption.split()

                    # Add \n every 10 words
                    caption = ""
                    for j, word in enumerate(words):
                        caption += word + " "
                        if j % 10 == 0 and j != 0:
                            caption += "\n"

                    save_image(inputs, caption, real_caption, solution_dir, batch_idx)

        # Compute metrics
        average_bleu_score = calculate_bleu(refs, hypos)
        cider_score = calculate_cider(refs, hypos)  
                
        print(f"Average BLEU score: {average_bleu_score:.4f}")
        print(f"CIDEr score: {cider_score:.4f}")
        print("Evaluation finished.")

        # Save the hypo and refs for the captions
        # as jsons

        with open(f"{captions_dir}/hypo.json", "w") as f:
            json.dump(hypos, f)
        
        with open(f"{captions_dir}/refs.json", "w") as f:
            json.dump(refs, f)
    
    else:

        # Load the hypo and refs for the captions
        # as jsons and compute BLEU and CIDEr scores
        # Only if the captions were already generated

        with open(f"{captions_dir}/hypo.json", "r") as f:
            hypos = json.load(f)
        
        with open(f"{captions_dir}/refs.json", "r") as f:
            refs = json.load(f)

        bleu_score = calculate_bleu(refs, hypos)
        cider_score = calculate_cider(refs, hypos)

        print(f"BLEU score: {bleu_score:.4f}")
        print(f"CIDEr score: {cider_score:.4f}")
        

if __name__ == "__main__":
    main()
