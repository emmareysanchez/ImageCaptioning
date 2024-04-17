import torch
from torch import nn
from torchvision import models
from src.data import Vocabulary


class ModifiedInception(nn.Module):
    def __init__(self, embedding_dim: int, aux_logits: bool = True):
        """
        Initialize the modified Inception model.

        Args:
            embedding_dim (int): The size of the embedding.

        """
        super(ModifiedInception, self).__init__()
        # Load the pretrained Inception model
        self.inception = models.inception_v3(pretrained=True, aux_logits=aux_logits)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.aux_logits = aux_logits

        # Freeze the parameters of the features for finetuning
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The extracted features.
        """
        # Directly handle whether to use logits or full outputs
        outputs = self.inception(images)
        if self.aux_logits and self.training:
            features = outputs.logits
        else:
            features = outputs  # This assumes the default is just the logits or a tensor

        # Apply dropout and ReLU to the processed features
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    """
    A Decoder RNN that generates captions from image features.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.5,
        pretrained_embedding=None,
    ):
        """
        Initialize the decoder RNN.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimension of word embeddings.
            hidden_dim (int): The dimension of the hidden state in the LSTM.
            num_layers (int): The number of layers in the LSTM.
            start_token_index (int): The index of the start token in the vocabulary.
            end_token_index (int): The index of the end token in the vocabulary.
            dropout (float): The dropout rate for regularization.
        """
        super(DecoderRNN, self).__init__()
        # Load word2vec pretrained embedding
        if pretrained_embedding:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass.

        Args:
            features (torch.Tensor): The image features extracted by the
            encoder.
            captions (torch.Tensor): The batch of captions (as word indices)
            for the images.

        Returns:
            torch.Tensor: The batch of predicted word indices for the captions.
        """
        embed = self.dropout(self.embedding(captions))

        # Add the image features to the caption embeddings like if they
        # were the first word in the sequence
        # Since we are passing the sentence without the last word,
        # dimmensions will match for the lstm
        new_embed = torch.cat((features.unsqueeze(0), embed), dim=0)

        # Pass the embeddings through the LSTM
        lstm_out, _ = self.lstm(new_embed)

        # Pass the LSTM outputs through the linear layer
        # to get the predicted word scores
        outputs = self.linear(lstm_out)
        return outputs


class ImageCaptioningModel(nn.Module):
    """
    An Image Captioning Model that combines an encoder and a decoder.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        num_layers: int,
        pretrained_embeddings=None,
    ):
        """
        Initialize the Image Captioning Model.

        Args:
            embedding_dim (int): The dimension of word embeddings.
            hidden_dim (int): The dimension of the hidden state in the LSTM.
            vocab_size (int): The size of the vocabulary.
            num_layers (int): The number of layers in the LSTM.
            start_token_index (int): The index of the start token in the vocabulary.
            end_token_index (int): The index of the end token in the vocabulary.
        """
        super(ImageCaptioningModel, self).__init__()
        self.encoder = ModifiedInception(embedding_dim)
        self.decoder = DecoderRNN(vocab_size,
                                  embedding_dim,
                                  hidden_dim,
                                  num_layers,
                                  pretrained_embedding=pretrained_embeddings)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass.

        Args:
            images (torch.Tensor): The batch of images.
            captions (torch.Tensor): The batch of captions (as word indices)
            for the images.

        Returns:
            torch.Tensor: The batch of predicted word indices for the captions.
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self,
                         image: torch.Tensor,
                         vocab: Vocabulary,
                         max_len: int = 50) -> str:
        """
        Generate a caption for a single image.

        Args:
            image (torch.Tensor): A single image tensor.
            vocab (Vocabulary): The Vocabulary object.
            max_len (int): Maximum length for the generated caption.

        Returns:
            str: The generated caption.
        """
        self.eval()
        caption = []
        with torch.no_grad():

            # Extract features from the image and stablish
            # the initial state for the lstm as None
            features = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_len):

                # Pass the features and the states through the lstm
                # to get the outputs and the new hidden state
                # and pass the outputs through the linear layer
                # to get the predicted word scores
                hidden, states = self.decoder.lstm(features, states)
                output = self.decoder.linear(hidden.squeeze(0))
                predicted = output.argmax(1)

                # Append the predicted word to the caption
                caption.append(predicted.item())

                # If the predicted word is the end token, stop
                if predicted == vocab.word2idx["</s>"]:
                    break

                # Get the embedding of the predicted word
                # to use it in the next iteration as the input
                features = self.decoder.embedding(predicted).unsqueeze(0)

        # Convert the predicted word indices to words
        return vocab.indices_to_caption(caption)
