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

    def forward(self, images: torch.Tensor, captions) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The predicted captions.
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image: torch.Tensor, vocab, max_len: int = 50) -> str:
        """
        Generate a caption for an image.

        Args:
            image (torch.Tensor): The input image.
            vocab (Vocab): The vocabulary object.
            max_len (int): The maximum length of the caption.

        Returns:
            str: The generated caption.
        """
        self.eval()
        with torch.no_grad():
            features = self.encoder(image).unsqueeze(0)
            start_token = vocab('<s>')
            end_token = vocab('</s>')

            caption = [start_token]

            for _ in range(max_len):
                captions = torch.tensor(caption).unsqueeze(0)
                captions = captions.long()
                outputs = self.decoder(features, captions)
                predicted = outputs.argmax(2)[-1].item()

                # Add the predicted word to the caption
        

                if predicted == end_token:
                    break

                caption.append(predicted)

    def generate_batch_captions(self, images: torch.Tensor, vocab, max_len: int = 50) -> list:
        """
        Generate captions for a batch of images.

        Args:
            images (torch.Tensor): The input images.
            vocab (Vocab): The vocabulary object.
            max_len (int): The maximum length of the caption.

        Returns:
            list: The generated captions.
        """
        self.eval()
        with torch.no_grad():
            features = self.encoder(images)
            start_token = vocab['<s>']
            end_token = vocab['</s>']

            captions = [[start_token] for _ in range(images.shape[0])]

            for _ in range(max_len):
                captions_tensor = torch.tensor(captions).long()
                outputs = self.decoder(features, captions_tensor)
                predicted = outputs.argmax(2)[:, -1]

                for i, token in enumerate(predicted):
                    if token.item() == end_token:
                        continue

                    captions[i].append(token.item())

            captions = [' '.join([vocab[token] for token in caption]) for caption in captions]

        return captions
        
