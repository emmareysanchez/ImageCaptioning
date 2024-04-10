# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final

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
                val_loader: DataLoader):
    
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define loss
    loss = torch.nn.MSELoss()

    # define tensorboard writer
    writer = SummaryWriter()

    # train model
    for epoch in range(epochs):
        train_step(model, train_loader, loss, optimizer, writer, epoch, device)
        val_step(model, val_loader, loss, writer, epoch, device)

    # save model
    save_model(model, "model")

