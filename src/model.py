# Deep learning libraries
import torch
from torch import nn
from torchvision import models

# Own modules
from src.data import Vocabulary

# To filter warnings
import warnings
warnings.filterwarnings("ignore")


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

        # Modify the output layer to fit the embedding size
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_dim)

        # Add a ReLU activation and a dropout layer
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
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)

        # Linear layer to get the predicted word scores and
        # the dropout layer to avoid overfitting
        self.linear = nn.Linear(hidden_dim, vocab_size)

        # Save the vocab size
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

        # Get the embeddings for the captions
        # to avoid overfitting
        embed = self.embedding(captions)

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

        # Initialize the encoder and the decoder
        self.encoder = ModifiedInception(embedding_dim)
        self.decoder = DecoderRNN(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            pretrained_embedding=pretrained_embeddings,
        )

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

        # Extract features from the images
        features = self.encoder(images)

        # Pass the features and the captions through the decoder
        outputs = self.decoder(features, captions)

        # Return the predicted word scores
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

        # Model to evaluation mode
        self.eval()

        # Initialize the caption list
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

    def generate_caption_beam_search(
        self,
        image: torch.Tensor,
        vocab: Vocabulary,
        beam_size: int = 20,
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
        with torch.no_grad():

            # Get first word (start token)
            hidden, states = self.decoder.lstm(self.encoder(image).unsqueeze(0), None)
            output = self.decoder.linear(hidden.squeeze(0))
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted = probs.argmax(1)

            # Get next words
            features = self.decoder.embedding(predicted).unsqueeze(0)
            hidden, states = self.decoder.lstm(features, states)
            output = self.decoder.linear(hidden.squeeze(0))
            probs = torch.nn.functional.softmax(output, dim=1)
            scores, words = probs.topk(beam_size)

            # normalize the scores
            scores = scores / scores.sum()

            # initialize the lists
            captions = []
            probabilities = []
            features_list = []
            states_list = []

            # get the k best captions
            for i in range(beam_size):
                predicted = torch.tensor([words[0, i].item()], device=image.device)
                probability = scores[0, i].item()
                captions.append([predicted.item()])
                probabilities.append(probability)
                features_list.append(self.decoder.embedding(predicted).unsqueeze(0))
                states_list.append(states)

            for j in range(2, max_len):

                # initialize the lists
                new_captions = []
                new_probabilities = []
                new_features_list = []
                new_states_list = []

                # for each caption in the beam, get the k best captions
                for i in range(beam_size):
                    # if the last word is the end token, stop
                    if captions[i][-1] == vocab.word2idx["</s>"]:
                        new_captions.append(captions[i])
                        new_probabilities.append(probabilities[i])
                        new_features_list.append(features_list[i])
                        new_states_list.append(states_list[i])

                    else:
                        # pass the features and the states through the lstm
                        hidden, states = self.decoder.lstm(
                            features_list[i], states_list[i]
                        )
                        output = self.decoder.linear(hidden.squeeze(0))
                        probs = torch.nn.functional.softmax(output, dim=1)
                        scores, words = probs.topk(beam_size)

                        # normalize the scores
                        scores = scores / scores.sum()

                        # get the k best captions and its probabilities
                        for k in range(beam_size):
                            predicted = torch.tensor(
                                [words[0, k].item()], device=image.device
                            )
                            new_probabilities.append(
                                probabilities[i]
                                * scores[0, k].item()
                                * (max_len - j)
                                / max_len
                            )
                            new_captions.append(captions[i] + [predicted.item()])
                            new_features_list.append(
                                self.decoder.embedding(predicted).unsqueeze(0)
                            )
                            new_states_list.append(states)

                # rank the captions by probability and get the k best
                best_captions = sorted(
                    list(
                        zip(
                            new_captions,
                            new_probabilities,
                            new_features_list,
                            new_states_list,
                        )
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )[:beam_size]
                captions = [x[0] for x in best_captions]
                probabilities = [x[1] for x in best_captions]
                features_list = [x[2] for x in best_captions]
                states_list = [x[3] for x in best_captions]

                # normalize the probabilities
                probabilities = [p / sum(probabilities) for p in probabilities]

        return vocab.indices_to_caption(captions[0])
