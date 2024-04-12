import torch

from src.model import MyModel
from src.train import train_model
from src.utils import load_data

# # Libraries to show the image and add the caption
# from PIL import Image
# import matplotlib.pyplot as plt

from src.utils import set_seed

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set all seeds
set_seed(42)

# static variables
DATA_PATH: str = "data"

need_to_train = True


def main():
    """
    This function is the main program. It downloads the data, preprocesses it,
    trains the model, and evaluates it.
    """
    # Define hyperparameters
    batch_size = 32
    num_epochs = 1
    learning_rate = 0.001
    embedding_size = 256
    hidden_size = 256
    num_layers = 1
    drop_prob = 0.5

    # load data
    (train_loader, val_loader, test_loader,
     word_to_index, index_to_word) = load_data(DATA_PATH, batch_size)

    if need_to_train:
        encoder_params = {'embedding_dim': embedding_size}
        decoder_params = {'vocab_size': len(word_to_index),
                          'embedding_dim': embedding_size,
                          'hidden_dim': hidden_size,
                          'num_layers': num_layers,
                          'start_token_index': word_to_index['<s>'],
                          'end_token_index': word_to_index['</s>'],
                          'dropout': drop_prob}
        model = MyModel(encoder_params, decoder_params)
        model.to(device)

        # train model
        train_model(device, num_epochs, learning_rate, model, train_loader, val_loader, word_to_index, index_to_word)

        # TODO: Implement the test of the model

    else:
        # We load the model from the models folder
        model = torch.load('models/model.pth')

    # XXX: See if this is necessary and if not, remove it
    # # We can now use the model to generate captions for new images
    # # Since images from the folder have already been preprocessed, we can use the
    # # following code to generate captions for them:
    # image = Image.open('path/to/image')
    # caption = model.generate_caption(image)
    # print(caption)

    # # We add the caption to the image and show it
    # plt.imshow(image)
    # plt.title(caption)
    # plt.show()


if __name__ == "__main__":
    main()
