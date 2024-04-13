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
            start_token = vocab("<s>")
            end_token = vocab("</s>")

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

    def generate_batch_captions(
        self, images: torch.Tensor, word2_idx, idx2_word, max_len: int = 50
    ) -> list:
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
            print("images", images.shape)  # images torch.Size([1, 3, 224, 224])
            features = self.encoder(images)
            print("features", features.shape)  # features torch.Size([1, 256])
            start_token = word2_idx["<s>"]
            end_token = word2_idx["</s>"]

            captions = [[start_token] for _ in range(images.shape[0])]

            for _ in range(max_len):
                captions_tensor = torch.tensor(captions).long()

                print(
                    "captions_tensor", captions_tensor.shape
                )  # captions_tensor torch.Size([1, 1])

                # to image device
                captions_tensor = captions_tensor.to(images.device)
                features = features.to(images.device)

                outputs = self.decoder(features, captions_tensor)
                print("outputs", outputs.shape)  # outputs torch.Size([1, 1, 7268])
                # the outputs are the probabilities of the next word
                # show the 10 most probable words
                _, indices = torch.topk(outputs, 10)

                # initialize predicted tensor
                predicted = torch.zeros(images.shape[0], 1).long()

                for i in range(indices.shape[0]):
                    # select the most probable word
                    predicted[i] = indices[i, 0, 1]
                    print([idx2_word[idx.item()] for idx in indices[i, 0, :]])

                for i, token in enumerate(predicted):
                    if token.item() == end_token:
                        break

                    captions[i].append(token.item())

            # remove the start token
            captions = [caption[1:] for caption in captions]

            # If the word is not in the dictionary we don't add it
            captions = [
                " ".join([idx2_word[token] for token in caption])
                for caption in captions
            ]

        return captions

    def load_model(self, name: str) -> None:
        """
        Load the model from a file.

        Args:
            path (str): The path to the file.
        """
        model = torch.jit.load(f"models/{name}.pt")
        self.load_state_dict(model.state_dict())
