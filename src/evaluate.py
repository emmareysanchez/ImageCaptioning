# deep learning libraries
import matplotlib.pyplot as plt
import torch

# other libraries
from typing import Final

# own modules
from src.utils import set_seed, load_data, load_checkpoint, save_image, download_embeddings
from src.model import ImageCaptioningModel

# TODO: Import necessary libraries
from PIL import Image
import numpy as np
import os

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
    embedding_size = 300
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
    with torch.no_grad():

        batch_idx = 0

        for inputs, targets in tqdm(test_loader):

            batch_idx += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Inputs must be float
            inputs = inputs.float()
            targets = targets.long()

            targets = targets.squeeze(1)
            real_caption = vocab.indices_to_caption(targets.tolist())

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
    print("Evaluation finished.")

if __name__ == "__main__":
    main()
