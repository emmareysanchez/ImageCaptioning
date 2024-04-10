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

    def forward(self, features, start_token_index, end_token_index, vocab_size):
        """
        Args:
            features (torch.Tensor): Tensor de características de imagen (vector de contexto) con shape (batch_size, feature_size).
            start_token_index (int): Índice del token de inicio en el vocabulario.
            end_token_index (int): Índice del token de fin en el vocabulario.
            vocab_size (int): Tamaño del vocabulario.

        Returns:
            torch.Tensor: Tensor con las log-probabilidades de las palabras predichas para cada posición en la secuencia.
        """
        batch_size = features.size(0)
        max_seq_length = 50 # FIXME: Verificar el máximo de palabras a generar

        # Inicializa el tensor para almacenar las log-probabilidades de las predicciones
        outputs = torch.zeros(batch_size, max_seq_length, vocab_size).to(features.device)

        # Inicializa el estado oculto y el estado de celda de la LSTM con el vector de contexto
        h, c = self.init_hidden(features)

        # Prepara el primer input para la LSTM, que será el token de inicio
        input_word = torch.full((batch_size,), start_token_index, dtype=torch.long).to(features.device)
        end_tokens = torch.full((batch_size,), end_token_index, dtype=torch.long).to(features.device)
        
        # Mascara para finalizar generación una vez se prediga el end token
        end_token_mask = torch.zeros(batch_size, dtype=torch.bool).to(features.device)

        for t in range(max_seq_length):
            # Embedding del input actual
            input_embedding = self.embedding(input_word).unsqueeze(1)
            
            # Pasa el embedding y el estado oculto a la LSTM
            lstm_out, (h, c) = self.lstm(input_embedding, (h, c))
            
            # Calcula la log-probabilidad de la siguiente palabra
            output = self.linear(lstm_out.squeeze(1))
            outputs[:, t, :] = output
            
            # Obtén la siguiente palabra (la de mayor log-probabilidad)
            input_word = output.argmax(1)
            
            # Comprobar si se ha predicho el end token y actualizar la máscara
            end_token_mask |= input_word == end_tokens
            
            # Si todas las secuencias han generado el end token, detener el bucle
            if end_token_mask.all():
                break

        return outputs
    
    def init_hidden(self, features):
        """
        Assuming features is the output of the CNN and has shape (batch_size, cnn_output_size),
        process it to be the initial hidden state of the LSTM (and cell state).
        
        Args:
            features (torch.Tensor): Output of the CNN with shape (batch_size, cnn_output_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the initial hidden state and cell state.
        """
        # Transform features to be the initial hidden state
        # Note: This might involve linear layers or other transformations,
        # depending on the dimensions needed.
        h0 = features.unsqueeze(0)  # Example transformation
        c0 = torch.zeros_like(h0)  # Initial cell state is often just zeros
        return h0, c0