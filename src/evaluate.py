# deep learning libraries
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final

# own modules
from src.utils import set_seed, load_model, generate_caption, load_data
from src.model import MyModel

# TODO: Import necessary libraries

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
    model = MyModel(encoder_params, decoder_params)
    model.load_model("model")
    # model = load_model("model")
    model = model.to(device)
    model.eval()

    # evaluate model
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Inputs must be float
            inputs = inputs.float()
            targets = targets.long()

            outputs = model(inputs, targets)

            # generate_caption
            # caption = model.generate_batch_captions(inputs, word_to_index, index_to_word)

            # Foreach output
            for i in range(len(outputs)):
                caption = generate_caption(outputs[i], index_to_word)
                real_caption = target_caption(targets[i], index_to_word)

                # show image with the cation
                image = inputs[0].cpu().numpy().transpose((1, 2, 0))
                plt.imshow(image)
                plt.title(f"Prediction: {caption}\nReal: {real_caption}")
                plt.show()


def target_caption(targets, index_to_word):
    """
    This function generates the target caption.

    Args:
        targets: The target caption.
        index_to_word: The index to word mapping.

    Returns:
        str: The target caption.
    """
    # Concat the words without the until the </s>
    caption = ""
    for i in targets:
        if i == index_to_word["</s>"]:
            break
        caption += index_to_word[i.item()] + " "
    return caption


if __name__ == "__main__":
    main()
