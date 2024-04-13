import torch
from torch import nn
from torchvision import models



class ModifiedVGG19(nn.Module):
    """
    A modified VGG19 model implemented using PyTorch for extracting image features.

    Attributes:
        features (nn.Sequential): The convolutional layers of the VGG19 model.
        classifier (nn.Sequential): The modified classifier layer.
    """
    def __init__(self, embedding_dim: int):
        """
        Initialize the modified VGG19 model.

        Args:
            embedding_dim (int): The size of the embedding.

        """
        super(ModifiedVGG19, self).__init__()
        # Load the pretrained VGG19 model
        vgg19 = models.vgg19(weights='DEFAULT').features

        # Here we keep the convolutional layers of VGG19 unchanged
        self.features = vgg19

        # Define the number of input features to the final linear layer.
        # For VGG19, this is 512 * 7 * 7 if using standard 224x224 input images.
        # This number may vary if you change the input size.
        num_features = 512 * 7 * 7

        # Replace the last FC layer to match the size of the embedding
        self.classifier = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Freeze the parameters of the features for finetuning
        for param in self.features.parameters():
            param.requires_grad = False

        # Optionally, you might decide to unfreeze some of these parameters later on.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The extracted features.
        """
        # Apply the convolutional layers
        x = self.features(x)

        # Flatten the features for the linear layer
        x = torch.flatten(x, 1)

        # Apply the modified classifier layer
        x = self.classifier(x)
        return x
