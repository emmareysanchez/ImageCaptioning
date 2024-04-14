# deep learning libraries
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final

# own modules
from src.utils import set_seed, load_model, generate_caption, load_data, generate_caption3
from src.model import MyModel, ImageCaptioningModel

# TODO: Import necessary libraries
from PIL import Image, ImageDraw, ImageFont
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
    (train_loader, val_loader, test_loader, word_to_index, index_to_word) = load_data(
        DATA_PATH, batch_size
    )
    encoder_params = {"embedding_dim": embedding_size}
    decoder_params = {
        "vocab_size": len(word_to_index),
        "embedding_dim": embedding_size,
        "hidden_dim": hidden_size,
        "num_layers": num_layers,
        "start_token_index": word_to_index["<s>"],
        "end_token_index": word_to_index["</s>"],
        "dropout": drop_prob,
    }
    # model = MyModel(encoder_params, decoder_params)
    model = ImageCaptioningModel(
        embedding_size,
        hidden_size,
        len(word_to_index),
        num_layers,
        word_to_index["<s>"],
        word_to_index["</s>"],
    )

    model.load_model("model")
    # model = load_model("model")

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

            # outputs = model(inputs, targets)

            # generate_caption
            # caption = model.generate_batch_captions(inputs, word_to_index, index_to_word)
            # caption = generate_caption3(model, inputs, index_to_word, word_to_index)

            # Foreach output
            for i in range(len(inputs)):
                # caption = generate_caption(outputs[i], index_to_word)
                # caption = generate_caption3(model, inputs[i], index_to_word, word_to_index)
                caption = model.generate_caption(inputs[i].unsqueeze(0), index_to_word)
                real_caption = target_caption(targets[i], index_to_word)


                # Save image and caption in "solution" folder
                # Will have to create it if necessary
                image = inputs[i].cpu().numpy().transpose((1, 2, 0))
                
                image = (image * 255).astype(np.uint8)  # Assuming image was normalized
                image = Image.fromarray(image)

                plt.figure()
                plt.imshow(image)
                plt.title(f"Predicted: {caption}\nReal: {real_caption}")
                plt.axis("off")
                plt.savefig(f"{solution_dir}/image_{batch_idx}_{i}.png")
                plt.close()


def target_caption(targets, index_to_word):
    """
    This function generates the target caption.

    Args:
        targets: The target caption.
        index_to_word: The index to word mapping.

    Returns:
        str: The target caption.
    """
    # Concat the words without the until the </s> and avoiding <s>
    caption = ""
    for i in targets[1:]:
        if index_to_word[i.item()] == "</s>":
            break
        caption += index_to_word[i.item()] + " "
    return caption



if __name__ == "__main__":
    main()
