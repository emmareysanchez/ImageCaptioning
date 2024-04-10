import torch
import torch.nn as nn

from encoder import ModifiedVGG19
from decoder import RNN

class MyModel(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        """
        Inicializa los componentes del modelo MyModel.

        Args:
            encoder_params (dict): Un diccionario con los parámetros necesarios para inicializar el encoder.
            decoder_params (dict): Un diccionario con los parámetros necesarios para inicializar el decoder.
        """
        super(MyModel, self).__init__()
        self.encoder = ModifiedVGG19(**encoder_params)
        self.decoder = RNN(**decoder_params)

    def forward(self, images, captions, caption_lengths):
        """
        Define el paso hacia adelante del modelo.

        Args:
            images (torch.Tensor): Tensor que contiene las imágenes de entrada.
            captions (torch.Tensor): Tensor que contiene las captions (subtítulos) de entrada asociadas a las imágenes.
            caption_lengths (List[int]): Lista que contiene las longitudes de las captions.

        Returns:
            torch.Tensor: El resultado del decoder.
        """
        features = self.encoder(images)
        outputs = self.decoder(captions, features, caption_lengths)
        return outputs
