import torch
from torch import nn
# import torch.nn.functional as F


class RNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) model implemented using PyTorch
    for generating image captions.

    Attributes:
        embedding (nn.Embedding): The embedding layer for the vocabulary.
        lstm (nn.LSTM): The LSTM layer.
        linear (nn.Linear): The fully connected layer to predict each word
        in the vocabulary.
        dropout (nn.Dropout): Dropout layer for regularization.
        vocab_size (int): Size of the vocabulary.
        start_token_index (int): Start token index in the vocabulary.
        end_token_index (int): End token index in the vocabulary.
    """

    def __init__(self, vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 start_token_index,
                 end_token_index,
                 dropout: float = 0.5):
        """
        Initialize the RNN model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the word embeddings.
            hidden_dim (int): Dimension of the hidden state of the LSTM.
            num_layers (int): Number of LSTM layers.
            start_token_index (int): Start token index in the vocabulary.
            end_token_index (int): End token index in the vocabulary.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        # Embedding layer for the vocabulary
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers,
                            # dropout=dropout,
                            batch_first=True)

        # Fully connected layer to predict each word in the vocabulary
        self.linear = nn.Linear(hidden_dim, vocab_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Save the vocab size
        self.vocab_size = vocab_size

        # Save the start and end token indices
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index

    def forward(self, features: torch.Tensor, captions) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Tensor of image features with shape
            (batch_size, feature_size).
            start_token_index (int): Start token index in the vocabulary.
            end_token_index (int):vEnd token index in the vocabulary.
            vocab_size (int): Size of the vocabulary.

        Returns:
            torch.Tensor: Tensor with the log-probabilities of the predicted
            words for each position in the sequence.
        """
        embed = self.dropout(self.embedding(captions))
        embed = torch.cat((features, embed), dim=1)
        lstm_out, _ = self.lstm(embed)
        outputs = self.linear(lstm_out)
        return outputs
    