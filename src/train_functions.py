# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import tqdm

from src.data import Vocabulary


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

        outputs = model(inputs, targets[:-1])

        outputs_reshaped = outputs.reshape(-1, outputs.shape[2])
        targets_reshaped = targets.reshape(-1)

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
    vocab: Vocabulary
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
        word2_idx (dict): dictionary to convert words to indexes.
        idx2_word (dict): dictionary to convert indexes to words.
    """
    model.to(device)
    # Model in evaluation mode
    model.eval()

    for inputs, targets in tqdm.tqdm(val_data):

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inputs must be float
        inputs = inputs.float()
        targets = targets.long()

        outputs = model(inputs, targets[:-1])

        outputs_reshaped = outputs.reshape(-1, outputs.shape[2])
        targets_reshaped = targets.reshape(-1)

        loss_value = loss(outputs_reshaped, targets_reshaped)

        writer.add_scalar("Loss/val", loss_value.item(), epoch)


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    data: DataLoader,
    device: torch.device,
    word2_idx: dict,
    idx2_word: dict
) -> np.ndarray:
    """
    This function predict the model.

    Args:
        model (torch.nn.Module): model to validate.
        data (DataLoader): dataloader of validation data.
        device (torch.device): device for running operations.
        word2_idx (dict): dictionary to convert words to indexes.
        idx2_word (dict): dictionary to convert indexes to words.

    Returns:
        np.ndarray: predictions of the model.
    """
    model.to(device)
    # Model in evaluation mode
    model.eval()

    predictions = []

    for inputs, targets in data:

        inputs = inputs.to(device)

        # Inputs must be float
        inputs = inputs.float()

        # FIXME: implement for more than one caption
        # get the captions of the images of the batch
        captions = model.genera_captions(inputs[0].unsqueeze(0), idx2_word)

        # Add the captions to the list of predictions
        predictions.extend(captions)

    # TODO: use and implement evaluation metrics to asses the predictions
