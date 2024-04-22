# ProyectoImageCaptioning

## Description
This project is about the implementation of a model that generates captions for images. The model is based on the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (2015). The model is implemented using PyTorch and trained on the Flickr30K dataset,
however, the model can be trained with the Flickr8K dataset as well. 

The model is based on a encoder-decoder architecture. The encoder is a pre-trained InceptionV3 model that extracts features from the images. The decoder is a LSTM network that generates the captions.

In addition, captions can be generated using a greedy aproach or a beam search aproach.

## Requirements
- numpy==1.26.2
- Pillow==10.1.0
- kaggle==1.6.11
- pandas==2.2.2
- tensorboard==2.15.1
- nltk==3.8.1
- rouge==1.0.1
- pycocoevalcap==1.2
- gensim==4.3.2
- scipy==1.12.0
- gdown==5.1.0
- mypy==1.7.0
- black==23.11.0
- flake8==7.0.0

In addition pytorch must be install following the instructions in the official website: https://pytorch.org/get-started/locally/


## Usage
Before running the code, you must have your Kaggle API credentials in a file called kaggle.json inside the folder ~/.kaggle of your computer user. You can get your credentials in your Kaggle account settings.

In addition, you must install the dependencies by running the following command:
```
pip install -r requirements.txt
```

To train the model, you must run the following command:
```
python -m src.train
```

Once a model is trained, you can generate captions and evaluate them with BLEU and CIDEr metrics by running the following command:
```
python -m src.evaluate
```

## Files
- src/train.py: Script to train the model. It includes the "need_to_load" variable that must be set to True if you want to load the model from a checkpoint or False if you want to train the model from scratch. The path to the checkpoint can be modified but by default it is set to "checkpoints/checkpoint.pth".

- src/evaluate.py: Script to generate captions and evaluate them with BLEU and CIDEr metrics. The path to the checkpoint can be modified but by default it is set to "checkpoints/checkpoint.pth". In addition, it includes 2 parameters that can be modified: "beam_search" and "debug". The "beam_search" parameter must be set to True if you want to use beam search to generate captions or False if you want to use a greedy approach. The "debug" parameter must be set to True if you want to print the generated captions with both approaches.

- src/utils.py: Script that contains utility functions to load the dataset, preprocess the images and captions, and evaluate the model.

- src/model.py: Script that contains the implementation of the model.

- src/data.py: Script that contains the implementation of the dataset, the Vocabulary class, and the CollateFn class.

- src/data_processing.py: Script that contains the implementation of the data processing functions.

- src/train_functions.py: Script that contains the training loop and validation loop functions.