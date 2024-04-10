import torch

from src.model import MyModel
from src.data import Flikr8kDataset
from src.train import train_model
from src.image_data_processing import download_and_prepare_flickr8k_dataset
from src.text_data_processing import load_and_process_captions_flickr8k, create_lookup_tables
from src.utils import load_data

# Libraries to show the image and add the caption
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
need_to_train = True

def main():
    
    path = 'data/Flicker8k_Dataset'

    # Define hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    embedding_size = 256
    hidden_size = 512
    num_layers = 1
    drop_prob = 0.5

    # load data
    # TODO: implementar función load_data que cree los datasets y los dataloaders
    train_loader, val_loader, test_loader, word_to_index, index_to_word = load_data(path, batch_size)


    if need_to_train:
        encoder_params = [embedding_size]
        decoder_params = [len(word_to_index), embedding_size, hidden_size, num_layers, drop_prob]
        model = MyModel(encoder_params, decoder_params)
        model.to(device)

        # train model
        train_model(device, num_epochs, learning_rate, model, train_loader, val_loader)

    else:
        # We load the model from the models folder
        model = torch.load('models/model.pth')

    # We can now use the model to generate captions for new images
    # Since images from the folder have already been preprocessed, we can use the following code
    # to generate captions for them
    # image = Image.open('path/to/image')
    # caption = model.generate_caption(image)
    # print(caption)

    # We add the caption to the image and show it
    # plt.imshow(image)
    # plt.title(caption)
    # plt.show()