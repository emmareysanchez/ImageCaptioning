# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional


@torch.enable_grad()
def train_step() -> None:
    """
    This function train the model.

    Args:
    """

    # TODO: Train step function


@torch.no_grad()
def val_step() -> None:
    """
    This function train the model.

    Args:
    """

    # TODO: Validation step function


@torch.no_grad()
def t_step() -> float:
    """
    This function tests the model.

    Args:

    Returns:
    """

    # TODO: Test step function
