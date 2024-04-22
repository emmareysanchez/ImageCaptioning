# Image Captioning using CNN-RNN Arquitecture

## Description
This project explores the intersection of deep learning and natural language processing (NLP) by implementing a model that generates captions for images. The model is based on the paper ["Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"](https://arxiv.org/abs/1502.03044) by Xu et al. (2015).

![Example image](captions/captions.png)

## Table of Contents
- [Image Captioning using CNN-RNN Arquitecture](#image-captioning-using-cnn-rnn-arquitecture)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Models](#models)
  - [Requirements](#requirements)
  - [Datasets](#datasets)
  - [Usage](#usage)
  - [Files](#files)

## Models
The project used a encoder-decoder architecture. The encoder is a CNN model and the decoder is a RNN model. The models used in this project are the following:
- [InceptionV3](https://pytorch.org/hub/pytorch_vision_inception_v3/): The CNN model is pre-trained on the ImageNet dataset. It is used to extract features from the images.
- [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html): The RNN model is used to generate the captions from the features extracted.

For generating captions, the model can use a greedy approach or a beam search approach. The beam search approach generates captions with better quality but it is slower than the greedy approach.

## Requirements
**Python 3.7** or higher must be installed in your computer.

The dependencies are listed in the `requirements.txt` file. All the dependencies can be installed by running the following command:
```
pip install -r requirements.txt
```

In addition **PyTorch** must be installed following the instructions in the official [website](https://pytorch.org/get-started/locally/).

## Datasets
The datasets used in this project were downloaded from Kaggle. A `kaggle.json` file must be created in the *~/.kaggle* folder of your computer user to download the datasets. You can get your credentials in your Kaggle account settings.
The datasets are the following:
- [Flickr8K](https://www.kaggle.com/adityajn105/flickr8k): The dataset contains 8,000 images with 5 captions each. Disk space required: 1 GB.
  
- [Flickr30K](https://www.kaggle.com/eeshawn/flickr30k): The dataset contains 31,000 images with 5 captions each. Disk space required: 4 GB.

## Usage

To train the model and download the datasets if they are not downloaded yet, you must run the following command:
```
python -m src.train
```

Once a model is trained, you can generate captions and evaluate them with BLEU and CIDEr metrics by running the following command:
```
python -m src.evaluate
```

## Files
- `src/train.py`: Script to train the model. It includes the "**need_to_load**" variable that must be set to *True* if you want to load the model from a checkpoint or *False* if you want to train the model from scratch. The path to the checkpoint can be modified but by default it is set to "checkpoints/checkpoint.pth".

- `src/evaluate.py`: Script to generate captions and evaluate them with BLEU and CIDEr metrics. The path to the checkpoint can be modified but by default it is set to "checkpoints/checkpoint.pth". In addition, it includes 2 parameters that can be modified: "**beam_search**" and "**debug**". The "**beam_search**" parameter must be set to *True* if you want to use beam search to generate captions or *False* if you want to use a greedy approach. The "debug" parameter must be set to *True* if you want to print the generated captions with both approaches.

- `src/utils.py`: Script that contains utility functions to load the dataset, preprocess the images and captions, and evaluate the model.

- `src/model.py`: Script that contains the implementation of the model.

- `src/data.py`: Script that contains the implementation of the dataset, the Vocabulary class, and the CollateFn class.

- `src/data_processing.py`: Script that contains the implementation of the data processing functions.

- `src/train_functions.py`: Script that contains the training loop and validation loop functions.