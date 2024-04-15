
# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter

# own modules
from src.utils import set_seed, save_checkpoint, load_checkpoint, load_data
from src.train_functions import train_step, val_step
from src.model import ImageCaptioningModel


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
    dataset_name = "flickr30k"  # "flickr8k" or "flickr30k"
    batch_size = 32
    epochs = 110
    learning_rate = 3e-4
    embedding_size = 256
    hidden_size = 256
    num_layers = 1

    checkpoint_save_path = CHECKPOINT_PATH + "_" + dataset_name + "_" + str(learning_rate)

    # load data
    (train_loader, val_loader, _,
     word_to_index, index_to_word) = load_data(DATA_PATH, dataset_name, batch_size)

    if need_to_train:
        # model = MyModel(encoder_params, decoder_params)
        model = ImageCaptioningModel(embedding_size,
                                     hidden_size,
                                     len(index_to_word),
                                     num_layers,
                                     word_to_index['<s>'],
                                     word_to_index['</s>']
                                     )
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
            start_epoch, model, optimizer = load_checkpoint(model,
                                                            optimizer,
                                                            CHECKPOINT_PATH)
            model.to(device)

        # train model showing progress
        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            # train_step(model, train_loader, loss, optimizer, writer, epoch, device)
            val_step(model, val_loader, loss, writer, epoch,
                     device, word_to_index, index_to_word)

            # Save a checkpoint into the checkpoint folder
            save_checkpoint(model, optimizer, epoch, 'checkpoint')

            # We save a checkpoint every 10 epochs into an specific folder
            # for the model
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, epoch, checkpoint_save_path, f"checkpoint_{epoch}")

        # Save the model into the models folder
        save_checkpoint(model, optimizer, epochs, "models", "model")
        print("Training finished.")

        # TODO: Implement the test of the model

    else:
        # We load the model from the models folder
        _, model, _ = load_checkpoint(model, None, "models", "model")


if __name__ == "__main__":
    main()
