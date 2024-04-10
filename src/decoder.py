import torch
from torch import nn
import torch.nn.functional as F

class RNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) model implemented using PyTorch for text classification.

    This model utilizes an embedding layer with pre-trained weights, followed by an LSTM layer
    for processing sequential data, and a linear layer for classification.

    Attributes:
        embedding (nn.Embedding): Embedding layer initialized with pre-trained weights.
        rnn (nn.LSTM): LSTM (Long Short Term Memory) layer for processing sequential data.
        fc (nn.Linear): Linear layer for classification.

    Args:
        embedding_weights (torch.Tensor): Pre-trained word embeddings.
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of layers in the LSTM.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.5):
        """
        Initializes the RNN model with given embedding weights, hidden dimension, and number of layers.

        Args:
            embedding_weights (torch.Tensor): The pre-trained embedding weights to be used in the embedding layer.
            hidden_dim (int): The size of the hidden state in the LSTM layer.
            num_layers (int): The number of layers in the LSTM.
        """
        super().__init__()
        # Embedding layer for the vocabulary
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        
        # Fully connected layer to predict each word in the vocabulary
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, captions: torch.Tensor, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            captions (torch.Tensor): Tensor de índices de palabras con shape (batch_size, max_length).
            features (torch.Tensor): Tensor de características de imagen con shape (batch_size, feature_size).
            lengths (torch.Tensor): Longitudes de cada caption en el lote, con shape (batch_size,).

        Returns:
            torch.Tensor: Salida del modelo con las predicciones de las próximas palabras, 
                           típicamente con shape (batch_size, max_length, vocab_size) después de aplicar el log_softmax.
        """

        # Embed caption inputs
        embeddings = self.dropout(self.embedding(captions)) # ¿Es necesario?

        # Pack the embedded captions
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        
        # Initialize the LSTM state with the CNN features
        # Here, 'features' must be shaped as (num_layers * num_directions, batch, hidden_size)
        # You might need to process 'features' to achieve this shape, depending on the CNN output
        h0, c0 = self.init_hidden(features)
        
        # Forward pass through LSTM
        packed_output, (hidden, _) = self.lstm(packed_embeddings, (h0, c0))
        
        # Unpack output
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Pass the output through the fully connected layer
        output = self.linear(self.dropout(output))
        
        return output
    
    def init_hidden(self, features):
        """Assuming features is the output of the CNN and has shape (batch_size, cnn_output_size),
        process it to be the initial hidden state of the LSTM (and cell state)."""
        batch_size = features.size(0)
        # Transform features to be the initial hidden state
        # Note: This might involve linear layers or other transformations,
        # depending on the dimensions needed.
        h0 = features.unsqueeze(0)  # Example transformation
        c0 = torch.zeros_like(h0)  # Initial cell state is often just zeros
        return h0, c0