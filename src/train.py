import torch

from src.model import ImageCaptioningModel
from src.utils import load_data

# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter

# own modules
from src.utils import set_seed, save_checkpoint, load_checkpoint
from src.train_functions import train_step, val_step


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set all seeds
set_seed(42)

# static variables
DATA_PATH: str = "data"
CHECKPOINT_PATH: str = "checkpoint"

need_to_train = True
need_to_load = False


def main():
    """
    This function is the main program. It downloads the data, preprocesses it,
    trains the model, and evaluates it.
    """
    # Define hyperparameters
    batch_size = 32
    epochs = 110
    learning_rate = 3e-4
    embedding_size = 256
    hidden_size = 256
    num_layers = 1

    # load data
    (train_loader, val_loader, _,
     word_to_index, index_to_word) = load_data(DATA_PATH, batch_size)

    if need_to_train:
        # model = MyModel(encoder_params, decoder_params)
        model = ImageCaptioningModel(embedding_size, hidden_size, len(word_to_index), num_layers,
                                     word_to_index['<s>'], word_to_index['</s>'])
        model.to(device)

        # define optimizer and loss ignoring the pad token
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss = torch.nn.CrossEntropyLoss(ignore_index=word_to_index['<PAD>'])

        # define tensorboard writer
        writer = SummaryWriter()

        # If needed, load the model from the checkpoint
        # that was saved during a previous training
        start_epoch = 0
        if need_to_load:
            start_epoch, model, optimizer = load_checkpoint(model, optimizer, CHECKPOINT_PATH)
            model.to(device)

        # train model showing progress
        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_step(model, train_loader, loss, optimizer, writer, epoch, device)
            val_step(model, val_loader, loss, writer, epoch, device, word_to_index, index_to_word)

            # Save a checkpoint
            save_checkpoint(model, optimizer, epoch, "checkpoint")

        print("Training finished.")

        # TODO: Implement the test of the model

    else:
        # We load the model from the models folder
        model = torch.load('models/model.pth')


if __name__ == "__main__":
    main()
