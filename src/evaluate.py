# deep learning libraries
from matplotlib import pyplot as plt
import torch

# other libraries
from typing import Final

# own modules
from src.utils import set_seed, load_data, load_checkpoint, target_caption
from src.model import ImageCaptioningModel

# TODO: Import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

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
    batch_size = 1
    embedding_size = 256
    hidden_size = 256
    num_layers = 1
    drop_prob = 0.5

    # load data
    (_, _, test_loader, word_to_index, index_to_word) = load_data(
        DATA_PATH, batch_size
    )

    # model = MyModel(encoder_params, decoder_params)
    model = ImageCaptioningModel(
        embedding_size,
        hidden_size,
        len(word_to_index),
        num_layers,
        word_to_index["<s>"],
        word_to_index["</s>"],
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

        for inputs, targets in test_loader:

            batch_idx += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Inputs must be float
            inputs = inputs.float()
            targets = targets.long()

            # Foreach output
            for i in range(len(inputs)):

                caption = model.generate_caption(inputs[i].unsqueeze(0), index_to_word)
                real_caption = target_caption(targets[i], index_to_word)

                # Save image and caption in "solution" folder
                # Will have to create it if necessary
                image = inputs[i].cpu().numpy().transpose((1, 2, 0))
                
                image = (image * 255).astype(np.uint8)  # Assuming image was normalized
                image = Image.fromarray(image)

                plt.figure()
                plt.imshow(image)
                plt.title(f"Predicted: {caption}\nReal: {real_caption}", fontsize=8)
                plt.axis('off')
                plt.tight_layout(pad=3.0)
                plt.savefig(f"{solution_dir}/image_{batch_idx}_{i}.png", dpi=300)
                plt.close()


if __name__ == "__main__":
    main()
