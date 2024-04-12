import torch
import torch.nn as nn

from src.encoder import ModifiedVGG19
from src.decoder import RNN


class MyModel(nn.Module):
    """
    A model that combines an encoder and a decoder for generating image captions.

    Attributes:
        encoder (ModifiedVGG19): The encoder model.
        decoder (RNN): The decoder model.
    """
    def __init__(self, encoder_params, decoder_params):
        """
        Initialize the model.

        Args:
            encoder_params (dict): Dictionary containing the parameters
            for the encoder.
            decoder_params (dict): Dictionary containing the parameters
            for the decoder.
        """
        super(MyModel, self).__init__()
        self.encoder = ModifiedVGG19(**encoder_params)
        self.decoder = RNN(**decoder_params)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The predicted captions.
        """
        features = self.encoder(images)

        print('MyModel features:', features.shape)
        outputs = self.decoder(features)
        return outputs
