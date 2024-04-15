# deep learning libraries
from torchvision import transforms

# other libraries
import os
import kaggle
import shutil
from PIL import Image

import pandas as pd


def download_and_prepare_dataset(path: str, dataset_name: str) -> None:
    """
    Download and prepare the Flickr8k or Flickr30k dataset from Kaggle
    and prepare it for training.

    Args:
        path (str): Path to save the processed data.
        dataset_name (str): Name of the dataset to download. Either
        "flickr8k" or "flickr30k".
    """

    # Kaggle dataset identifier
    if dataset_name == "flickr8k":
        dataset_identifier: str = "adityajn105/flickr8k"
    else:
        dataset_identifier: str = "eeshawn/flickr30k"

    # Make sure the kaggle.json file is set up and permissions are correct
    kaggle.api.authenticate()

    # Create path if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    if dataset_name == "flickr8k":
        dataset_path = f"{path}/flickr8k"
    else:
        dataset_path = f"{path}/flickr30k"

    # Download dataset
    # Only download the dataset if it hasn't been downloaded yet
    if not os.path.exists(dataset_path):
        kaggle.api.dataset_download_files(dataset_identifier,
                                          path=dataset_path,
                                          unzip=True)

        # Prepare directories for processed data
        if not os.path.exists(f"{dataset_path}/train"):
            os.makedirs(f"{dataset_path}/train")
        if not os.path.exists(f"{dataset_path}/val"):
            os.makedirs(f"{dataset_path}/val")
        if not os.path.exists(f"{dataset_path}/test"):
            os.makedirs(f"{dataset_path}/test")

        # For inception dataset
        transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
            ]
        )

        if dataset_name == "flickr8k":
            images_path = f"{dataset_path}/Images"
        else:
            images_path = f"{dataset_path}/flickr30k_images"

        images_list = os.listdir(images_path)

        # Split into train and validation
        # 80% train, 20% validation
        test_images = images_list[int(len(images_list) * 0.8):]
        train_images = images_list[: int(len(images_list) * 0.8)]

        # Of the train images, 80% will be used for training and 20% for validation
        val_images = train_images[int(len(train_images) * 0.8):]
        train_images = train_images[: int(len(train_images) * 0.8)]

        # Process and save images
        list_splits = ["train", "val", "test"]
        list_class_dirs = [train_images, val_images, test_images]

        for i in range(len(list_splits)):

            split = list_splits[i]
            list_images = list_class_dirs[i]

            for image_file in list_images:
                image_path = f"{images_path}/{image_file}"
                image = Image.open(image_path).convert("RGB")
                image = transform(image)
                image.save(f"{dataset_path}/{split}/{image_file}")

        shutil.rmtree(images_path)

    print("Dataset processed and saved.")


def divide_captions(path: str) -> None:
    """
    Divides the captions into train, validation and
    test sets. It saves the result in a captions, txt file.

    Args:
        path (str): path where the data was downloaded.
    """
    # Open the captions.txt file as a DataFrame
    captions_path = path + "/captions.txt"

    # Separate the images into train, validation and test sets
    # Only if the images are not already separated
    if os.path.exists(captions_path):

        df_captions = pd.read_csv(captions_path, sep=',')

        # If it's the Flickr30k dataset, change the column names to match
        # the Flickr8k dataset
        if 'image' not in df_captions.columns:
            df_captions = df_captions.drop(columns=['comment_number'])
            df_captions.columns = ['image', 'caption']

        images_train = os.listdir(path + '/train')
        images_val = os.listdir(path + '/val')
        images_test = os.listdir(path + '/test')

        # Filter the captions for each set
        df_captions_train = df_captions[df_captions['image'].isin(images_train)]
        df_captions_val = df_captions[df_captions['image'].isin(images_val)]
        df_captions_test = df_captions[df_captions['image'].isin(images_test)]

        # Save the captions for each set in a txt file
        df_captions_train.to_csv(path + "/captions_train.txt", index=False)
        df_captions_val.to_csv(path + "/captions_val.txt", index=False)
        df_captions_test.to_csv(path + "/captions_test.txt", index=False)

        # Remove the captions.txt file
        os.remove(os.path.join(path, 'captions.txt'))

    print("Captions divided and saved.")
