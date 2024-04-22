# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import tqdm


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model (torch.nn.Module): model to train.
        train_data (DataLoader): dataloader of training data.
        loss (torch.nn.Module): loss function.
        optimizer (torch.optim.Optimizer): optimizer.
        writer (SummaryWriter): writer for tensorboard.
        epoch (int): epoch of the training.
        device (torch.device): device for running operations.
    """

    # Model to device
    model.to(device)

    # Model in training mode
    model.train()

    # Lists to store losses
    losses_train = []

    for _, inputs, targets in tqdm.tqdm(train_data):

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inputs must be float
        inputs = inputs.float()
        targets = targets.long()

        optimizer.zero_grad()

        outputs = model(inputs, targets[:-1])

        # Reshape outputs and targets to calculate the loss
        outputs_reshaped = outputs.reshape(-1, outputs.shape[2])
        targets_reshaped = targets.reshape(-1)

        loss_value = loss(outputs_reshaped, targets_reshaped)

        loss_value.backward()
        optimizer.step()

        losses_train.append(loss_value.item())

    loss_mean = sum(losses_train) / len(losses_train)
    writer.add_scalar("Loss/train", loss_mean, epoch)


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device
) -> None:
    """
    This function validate the model.

    Args:
        model (torch.nn.Module): model to validate.
        val_data (DataLoader): dataloader of validation data.
        loss (torch.nn.Module): loss function.
        writer (SummaryWriter): writer for tensorboard.
        epoch (int): epoch of the validation.
        device (torch.device): device for running operations.
    """

    # Model to device
    model.to(device)

    # Model in evaluation mode
    model.eval()

    # To save the losses
    losses_val = []

    for _, inputs, targets in tqdm.tqdm(val_data):

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inputs must be float
        inputs = inputs.float()
        targets = targets.long()

        outputs = model(inputs, targets[:-1])

        # Reshape outputs and targets to calculate the loss
        outputs_reshaped = outputs.reshape(-1, outputs.shape[2])
        targets_reshaped = targets.reshape(-1)

        loss_value = loss(outputs_reshaped, targets_reshaped)

        losses_val.append(loss_value.item())

    loss_mean = sum(losses_val) / len(losses_val)
    writer.add_scalar("Loss/val", loss_mean, epoch)
