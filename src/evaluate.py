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

    solution_dir = "solution"
    if not os.path.exists(solution_dir):
        os.makedirs(solution_dir)

    model = model.to(device)
    model.eval()

    # evaluate model
    
    bleu_scores =[]
    with torch.no_grad():

        batch_idx = 0
        bleu_for_img = 0

        for inputs, targets in tqdm(test_loader):

            batch_idx += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Inputs must be float
            inputs = inputs.float()
            targets = targets.long()

            targets = targets.squeeze(1)
            real_caption = vocab.indices_to_caption(targets.tolist())
            
            refs = defaultdict(list)
            hypos = {}
         
            img_id = f'image_{batch_idx}'
            refs[img_id].append(real_caption)
            hypos[img_id] = caption

            if batch_idx % 5 == 0:

                # Only generate the caption ones for the five images
                # that are the same
                caption = model.generate_caption(inputs, vocab)

                words = caption.split()

                # Add \n every 10 words
                caption = ""
                for j, word in enumerate(words):
                    caption += word + " "
                    if j % 10 == 0 and j != 0:
                        caption += "\n"
                        
                

                save_image(inputs, caption, real_caption, solution_dir, batch_idx)

            # TODO: implementar métricas de error
            cider_score = calculate_cider(refs, hypos)
            bleu_score = calculate_bleu([real_caption], caption)
            if bleu_score > bleu_for_img:
                bleu_for_img = bleu_score
            
            if batch_idx % 5 == 4:
                bleu_scores.append(bleu_for_img)
                bleu_for_img = 0
            # bleu_scores.append(bleu_score)
        
        average_bleu_score = sum(bleu_scores) / len(bleu_scores)
            
            
    print(f"Average BLEU score: {average_bleu_score:.4f}")
    print(f"CIDEr score: {cider_score:.4f}")
    print("Evaluation finished.")

if __name__ == "__main__":
    main()
