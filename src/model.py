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
            features = (
                outputs  # This assumes the default is just the logits or a tensor
            )

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
        self.decoder = DecoderRNN(vocab_size, embedding_dim, hidden_dim, num_layers)

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

    def generate_caption(
        self, image: torch.Tensor, vocab: Vocabulary, max_len: int = 50
    ) -> str:
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

    def beam_search(self, beam, branch_factor, max_depth, depth, beam_index):
        """
        Perform the process of beam search.

        Args:
            beam (torch.Tensor): The beam tensor.
            branch_factor (int): The branch factor.
            max_depth (int): The maximum depth.
            depth (int): The current depth.
            beam_index (int): The current index in the beam tensor.

        Returns:
            torch.Tensor: The updated beam tensor.
        """
        if max_depth == 0:
            return beam

        for i in range(branch_factor):
            beam_index += i * (branch_factor ** (max_depth - 1))

            # get features and states
            for j in range(depth):
                word = beam[beam_index, j, 0]
                features = self.encoder.embedding(word).unsqueeze(0)
                hidden, states = self.decoder.lstm(features, None)

            # get the output and the scores
            output = self.decoder.linear(hidden.squeeze(0))
            words, scores = output.topk(branch_factor)

            # update the beam tensor
            for j in range(branch_factor):
                beam[beam_index + j, depth, 0] = words[0, j]
                beam[beam_index + j, depth, 1] = scores[0, j]

            beam = self.beam_search(
                beam, branch_factor, max_depth - 1, depth + 1, beam_index
            )
        return beam

    def generate_caption_beam_search(
        self,
        image: torch.Tensor,
        vocab: Vocabulary,
        branch_factor: int = 4,
        max_depth: int = 4,
        max_len: int = 50,
    ) -> str:
        """
        Generate a caption for a single image using beam search.

        Args:
            image (torch.Tensor): A single image tensor.
            vocab (Vocabulary): The Vocabulary object.
            beam_size (int): The size of the beam.
            max_len (int): Maximum length for the generated caption.

        Returns:
            str: The generated caption.
        """
        self.eval()
        # hacer un tensor de ceros con el tamaño de (branch factor ^ max_len)xmax_lenx2
        beam = torch.zeros(branch_factor**max_depth, max_len, 2)
        with torch.no_grad():

            # Extract features from the image and stablish
            # the initial state for the lstm as None
            features = self.encoder(image).unsqueeze(0)
            states = None
            # Iterate over the maximum length jumping by the branch factor
            for i in range(0, max_len, branch_factor):
                if i == 0:
                    for j in range(branch_factor):
                        hidden, states = self.decoder.lstm(features, states)
                        output = self.decoder.linear(hidden.squeeze(0))
                        words, scores = output.topk(branch_factor)
                        beam_index = j * (branch_factor ** (max_depth - 1))
                        beam[beam_index, 0, 0] = words[0, j]
                        beam[beam_index, 0, 1] = scores[0, j]
                else:
                    # Get the top k predicted captions from the beam
                    new_beam = torch.zeros(branch_factor**max_depth, max_len, 2)
                    beam_rank = beam[:, i - 1, 1].argsort(descending=True)
                    for j in range(branch_factor):
                        # Get the k best captions and append it to the new beam tensor
                        beam_index = j * (branch_factor ** (max_depth - 1))
                        new_beam[beam_index, :i, :] = beam[beam_rank[j], :i, :]
                    beam = new_beam

                beam = self.beam_search(beam, branch_factor, max_depth, i, 0)

        # Get the best caption from the beam
        beam_rank = beam[:, -1, 1].argsort(descending=True)

        best_captions = []
        # Get the k best captions to str
        for i in range(branch_factor):
            caption = []
            for j in range(max_len):
                word = beam[beam_rank[i], j, 0]
                caption.append(word.item())
                if word == vocab.word2idx["</s>"]:
                    break
            best_captions.append(vocab.indices_to_caption(caption))
            print(best_captions)

        return best_captions[0]
