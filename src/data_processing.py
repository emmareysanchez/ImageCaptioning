# deep learning libraries
from torchvision import transforms

# other libraries
import os
import kaggle
import shutil
from PIL import Image

import pandas as pd
import json



def download_and_prepare_flickr8k_dataset(path: str) -> None:
    """
    Download and prepare the Flickr8k dataset from Kaggle and process it.

    Args:
        path (str): Path to save the processed data.
    """

    # Kaggle dataset identifier
    dataset_identifier: str = "adityajn105/flickr8k"

    # Make sure the kaggle.json file is set up and permissions are correct
    kaggle.api.authenticate()

    # Create path if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    dataset_path = f"{path}/flickr8k"

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

        images_list = os.listdir(f"{dataset_path}/Images")

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

            # Adjust according to the actual Flickr8k structure on disk
            images_path = f"{dataset_path}/Images"

            for image_file in list_images:
                image_path = f"{images_path}/{image_file}"
                image = Image.open(image_path).convert("RGB")
                image = transform(image)
                image.save(f"{dataset_path}/{split}/{image_file}")

        shutil.rmtree(f"{dataset_path}/Images")

    print("Dataset processed and saved.")


def organize_caption_flickr8k(path: str) -> None:
    """"
    Organize the captions from the Flickr8k dataset into JSON files for
    each set.

    Args:
        path (str): path where the data was downloaded.
    """
    # Open the captions.txt file as a DataFrame
    captions_path = path + "/captions.txt"

    # Separate the images into train, validation and test sets
    # Only if the images are not already separated
    if os.path.exists(captions_path):
        df_captions = pd.read_csv(captions_path, sep=',')

        images_train = os.listdir(path + '/train')
        images_val = os.listdir(path + '/val')
        images_test = os.listdir(path + '/test')

        # Filter the captions for each set
        df_captions_train = df_captions[df_captions['image'].isin(images_train)]
        df_captions_val = df_captions[df_captions['image'].isin(images_val)]
        df_captions_test = df_captions[df_captions['image'].isin(images_test)]

        # Create JSON for each set
        sets = {'train': df_captions_train,
                'val': df_captions_val,
                'test': df_captions_test}
        for set_name, set_captions in sets.items():
            # Group captions by image and convert to dictionary
            image_captions = (set_captions.groupby('image')['caption']
                              .apply(list).to_dict())

            # Write JSON file
            with open(os.path.join(path, f'captions_{set_name}.json'), 'w') as json_file:
                json.dump(image_captions, json_file, indent=4)

        # Remove the captions.txt file
        os.remove(os.path.join(path, 'captions.txt'))


def divide_captions_flickr8k(path: str) -> None:
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
