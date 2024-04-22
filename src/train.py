# Deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter

# Own modules
from src.utils import (set_seed,
                       save_checkpoint,
                       load_checkpoint,
                       load_data,
                       download_embeddings)
from src.train_functions import train_step, val_step
from src.model import ImageCaptioningModel


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set all seeds
set_seed(42)

# Static variables
DATA_PATH: str = "data"
CHECKPOINT_PATH: str = "checkpoint"


need_to_load = True


def main() -> None:
    """
    This function is the main program. It downloads the data, preprocesses it,
    trains the model, and evaluates it.
    """

    # Define hyperparameters
    dataset_name = "flickr30k"  # "flickr8k" or "flickr30k"
    batch_size = 264
    epochs = 110
    learning_rate = 3e-4
    embedding_size = 300
    hidden_size = 256
    num_layers = 1

    checkpoint_save_path = CHECKPOINT_PATH + "_" + dataset_name +\
        "_" + str(learning_rate)

    # load data
    (train_loader, val_loader, _,
     vocab) = load_data(DATA_PATH, dataset_name, batch_size)

    if need_to_load:

        # Define the type of model as well as the optimizer

        model_type = ImageCaptioningModel(embedding_size,
                                          hidden_size,
                                          len(vocab),
                                          num_layers
                                          )
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model_type.parameters(),
                                                            lr=learning_rate
                                                            )

        # Load the model from the checkpoint
        last_epoch, model, optimizer_loaded = load_checkpoint(model_type,
                                                              optimizer,
                                                              CHECKPOINT_PATH
                                                              )
        model.to(device)

        if optimizer_loaded is not None:
            # optimizer_loaded params must be in the same device
            # as the optimizer
            for state in optimizer_loaded.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            optimizer = optimizer_loaded

        # Define the start epoch
        start_epoch = last_epoch + 1

    else:

        # Load the pretrained embeddings
        word2vec = download_embeddings()
        pretrained_embeddings = vocab.load_pretrained_embeddings(word2vec,
                                                                 embedding_size)

        # Define the model with the pretrained embeddings
        model = ImageCaptioningModel(embedding_size,
                                     hidden_size,
                                     len(vocab),
                                     num_layers,
                                     pretrained_embeddings=pretrained_embeddings
                                     )
        model.to(device)

        # Define optimizer and loss ignoring the pad token
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Define the start epoch
        start_epoch = 0

    loss = torch.nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])

    # define tensorboard writer
    writer = SummaryWriter()

    # train model showing progress
    for epoch in range(start_epoch, epochs):

        print(f"Epoch {epoch + 1}/{epochs}")

        # Perform training and validation steps
        train_step(model,
                   train_loader,
                   loss,
                   optimizer,
                   writer,
                   epoch,
                   device)
        val_step(model,
                 val_loader,
                 loss,
                 writer,
                 epoch,
                 device)

        # Save a checkpoint into the checkpoint folder
        save_checkpoint(model, optimizer, epoch, 'checkpoint')

        # We save a checkpoint every 10 epochs into an specific folder
        # for the model
        if epoch % 10 == 0:
            save_checkpoint(model,
                            optimizer,
                            epoch,
                            checkpoint_save_path,
                            f"checkpoint_{epoch}")

    # Save the model into the models folder
    save_checkpoint(model, optimizer, epochs, "models", "model")
    print("Training finished.")


if __name__ == "__main__":
    main()
