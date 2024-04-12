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
                            dropout=dropout,
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
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
        batch_size = features.size(0)
        max_seq_length = 50  # FIXME: Verify the maximum sequence length

        # Initialize the tensor to store the log-probabilities of the predicted words
        outputs = torch.zeros(batch_size,
                              max_seq_length,
                              self.vocab_size).to(features.device)

        # Initialize the hidden state and cell state of the LSTM with the image features
        h, c = self.init_hidden(features)

        # Prepare the first input for the LSTM, which will be the start token
        input_word = torch.full((batch_size,),
                                self.start_token_index,
                                dtype=torch.long).to(features.device)
        end_tokens = torch.full((batch_size,),
                                self.end_token_index,
                                dtype=torch.long).to(features.device)

        # Initialize a mask to keep track of which sequences have generated the end token
        end_token_mask = torch.zeros(batch_size, dtype=torch.bool).to(features.device)

        for t in range(max_seq_length):
            # Embed the input word
            input_embedding = self.embedding(input_word).unsqueeze(1)

            # Apply the LSTM layer
            lstm_out, (h, c) = self.lstm(input_embedding, (h, c))

            # Apply the linear layer to get the log-probabilities of the predicted words
            output = self.linear(lstm_out.squeeze(1))
            outputs[:, t, :] = output

            # Apply dropout and select the word with the highest log-probability
            input_word = output.argmax(1)

            # Update the end token mask
            end_token_mask |= input_word == end_tokens

            # If all sequences have generated the end token, stop the loop
            if end_token_mask.all():
                break

        return outputs

    def init_hidden(self, features):
        """
        Assuming features is the output of the CNN and has shape
        (batch_size, cnn_output_size), process it to be the initial
        hidden state of the LSTM (and cell state).

        Args:
            features (torch.Tensor): Output of the CNN with shape (batch_size,
            cnn_output_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the initial
            hidden state and cell state.
        """
        # Transform features to be the initial hidden state
        # Note: This might involve linear layers or other transformations,
        # depending on the dimensions needed.
        h0 = features.unsqueeze(0)  # Add a dimension for the sequence length
        c0 = torch.zeros_like(h0)  # Initial cell state is often just zeros
        return h0, c0
