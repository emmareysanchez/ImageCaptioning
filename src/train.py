# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# own modules
from src.utils import set_seed, save_model
from src.train_functions import train_step, val_step



# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def train_model(device: torch.device,
                epochs: int,
                lr: float,
                model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                word2_idx: dict,
                idx2_word: dict):
    """
    This function trains the model.

    Args:
        device (torch.device): device to use.
        epochs (int): number of epochs.
        lr (float): learning rate.
        model (torch.nn.Module): model to train.
        train_loader (DataLoader): train data loader.
        val_loader (DataLoader): validation data loader.
    """

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define Cross Entropy loss ignoring the pad token
    loss = torch.nn.CrossEntropyLoss(ignore_index=word2_idx['<PAD>'])

    # define tensorboard writer
    writer = SummaryWriter()

    # train model showing progress
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_step(model, train_loader, loss, optimizer, writer, epoch, device)
        val_step(model, val_loader, loss, writer, epoch, device, word2_idx, idx2_word)

    # save model
    save_model(model, "model")
