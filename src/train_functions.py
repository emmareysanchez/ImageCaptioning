# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import tqdm

# # other libraries
# from typing import Optional
# from src.utils import generate_caption


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
    model.to(device)
    # Model in training mode
    model.train()

    for inputs, targets in tqdm.tqdm(train_data):

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inputs must be float
        inputs = inputs.float()
        targets = targets.long()

        optimizer.zero_grad()

        outputs = model(inputs, targets)

        # print(outputs.shape)
        # print(targets.shape)

        outputs_reshaped = outputs.reshape(-1, outputs.shape[2])
        targets_reshaped = targets.reshape(-1)

        # print(outputs_reshaped.shape)
        # print(targets_reshaped.shape)

        loss_value = loss(outputs_reshaped, targets_reshaped)

        loss_value.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss_value.item(), epoch)
        


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    word2_idx: dict,
    idx2_word: dict
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
    model.to(device)
    # Model in evaluation mode
    model.eval()

    for inputs, targets in val_data:

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inputs must be float
        inputs = inputs.float()
        targets = targets.long()

        # We generate the caption for the batch
        captions = model.generate_batch_captions(inputs,word2_idx, idx2_word)
        print(captions)

        # We compute the loss
        outputs = model(inputs, targets)
        outputs_reshaped = outputs.reshape(-1, outputs.shape[2])
        targets_reshaped = targets.reshape(-1)
        loss_value = loss(outputs_reshaped, targets_reshaped)

        writer.add_scalar("Loss/val", loss_value.item(), epoch)


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    data: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    This function predict the model.

    Args:
        model (torch.nn.Module): model to validate.
        data (DataLoader): dataloader of validation data.
        device (torch.device): device for running operations.

    Returns:
        np.ndarray: predictions of the model.
    """
    model.to(device)
    # Model in evaluation mode
    model.eval()

    predictions = []

    for inputs, _ in data:

        inputs = inputs.to(device)

        # Inputs must be float
        inputs = inputs.float()

        outputs = model(inputs)

        predictions.append(outputs.cpu().numpy())

    # TODO: use and implement evaluation metrics to asses the predictions
