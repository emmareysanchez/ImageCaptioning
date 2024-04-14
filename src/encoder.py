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


class ModifiedInception(nn.Module):
    def __init__(self, embedding_dim: int):
        """
        Initialize the modified Inception model.

        Args:
            embedding_dim (int): The size of the embedding.

        """
        super(ModifiedInception, self).__init__()
        # Load the pretrained Inception model
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

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
        features = self.inception(images)
        if self.training and self.inception.aux_logits:
            features = features.logits  # Solo si aux_logits está habilitado y es relevante para tu uso.
        else:
            features = features  # O maneja adecuadamente según tu caso de uso.
        return self.dropout(self.relu(features))