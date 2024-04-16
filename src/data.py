# deep learning libraries
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# other libraries
from PIL import Image

import pandas as pd


class Vocabulary:
    def __init__(self, freq_threshold: int) -> None:
        """
        Initialize the Vocabulary object.

        Args:
            freq_threshold (int): The frequency threshold for including
            a word in the vocabulary.
        """
        self.idx2word = {0: "<PAD>",
                         1: "<s>",
                         2: "</s>",
                         3: "<UNK>"}
        self.word2idx = {"<PAD>": 0,
                         "<s>": 1,
                         "</s>": 2,
                         "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self) -> int:
        """
        Return the length of the vocabulary.
        """
        return len(self.idx2word)

    @staticmethod
    def tokenizer(text: str, extra_tokens: bool = False) -> list:
        """
        Tokenize the text.
        """

        text = text.lower().replace('"', '').replace("'", '')

        if text[-1] == '.':
            text = text[:-1]

        text = text.strip()
        text = " ".join(text.split())
        text = "<s> " + text + " </s>"

        if extra_tokens:
            text = text.replace(".", " <PERIOD> ")
            text = text.replace(",", " <COMMA> ")
            text = text.replace('"', " <QUOTATION_MARK> ")
            text = text.replace(";", " <SEMICOLON> ")
            text = text.replace("!", " <EXCLAMATION_MARK> ")
            text = text.replace("?", " <QUESTION_MARK> ")
            text = text.replace("(", " <LEFT_PAREN> ")
            text = text.replace(")", " <RIGHT_PAREN> ")
            text = text.replace("--", " <HYPHENS> ")
            text = text.replace("?", " <QUESTION_MARK> ")
            text = text.replace(":", " <COLON> ")
        return text.split(" ")

    @staticmethod
    def untokenizer(tokens: list) -> str:
        """
        Untokenize the tokens.
        """
        text = " ".join(tokens)
        text = text.replace(" <PERIOD> ", ".")
        text = text.replace(" <COMMA> ", ",")
        text = text.replace(" <QUOTATION_MARK> ", '"')
        text = text.replace(" <SEMICOLON> ", ";")
        text = text.replace(" <EXCLAMATION_MARK> ", "!")
        text = text.replace(" <QUESTION_MARK> ", "?")
        text = text.replace(" <LEFT_PAREN> ", "(")
        text = text.replace(" <RIGHT_PAREN> ", ")")
        text = text.replace(" <HYPHENS> ", "--")
        text = text.replace(" <QUESTION_MARK> ", "?")
        text = text.replace(" <COLON> ", ":")
        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
        return text

    def build_vocabulary(self, sentences: list):
        """
        Build the vocabulary with the words that appear
        at least `freq_threshold` times in the sentences.
        """
        frequencies = {}
        idx = len(self.idx2word)
        for sentence in sentences:
            for word in self.tokenizer(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if (frequencies[word] == self.freq_threshold
                   and word not in self.word2idx):
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def caption_to_indices(self, caption: str) -> list:
        """
        Convert a caption to a list of word indices.
        """
        tokens = self.tokenizer(caption, extra_tokens=True)
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def indices_to_caption(self, indices: list) -> str:
        """
        Convert a list of word indices to a caption.
        Stop when the end token is found.
        """
        # Translate the indices to words and stop when the end token is found
        tokens = []
        for idx in indices:
            tokens.append(self.idx2word[idx])
            if idx == self.word2idx["</s>"]:
                break
        return self.untokenizer(tokens)


class ImageAndCaptionsDataset(Dataset):
    """
    Dataset class for the Image and Captions dataset.

    Attributes:
        captions_path (str): path to the captions file.
        images_path (str): path to the images folder.
        df (pd.DataFrame): dataframe with the captions.
        transform (callable): transform to apply to the images.
        image_names (pd.Series): series with the image names.
        vocab (Vocabulary): vocabulary object.
    """
    def __init__(self, captions_path: str, images_path: str, transform=None, vocab=None):
        """
        Initialize the ImageAndCaptionsDataset object.

        Args:
            captions_path (str): path to the captions file.
            images_path (str): path to the images folder.
            transform (callable): transform to apply to the images.
            vocab (Vocabulary): vocabulary object.
        """
        self.captions_path = captions_path
        self.images_path = images_path

        self.df = pd.read_csv(captions_path)
        self.transform = transform

        self.image_names = self.df["image"]
        self.captions = self.df["caption"]

        if vocab:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(freq_threshold=5)
            self.vocab.build_vocabulary(self.captions.tolist())

        print(f"Vocabulary size: {len(self.vocab)}")

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.image_names)

    def __getitem__(self, index):
        """
        Get the item at the specified index.

        Args:
            index (int): index of the item.

        Returns:
            tuple: tuple with the image and the tokenized caption.
        """
        # Load image path and captions
        image_path = self.image_names[index]
        caption = self.captions[index]

        # Load image transforming it to tensor
        image = Image.open(self.images_path + "/" + image_path)
        if self.transform:
            image = self.transform(image)

        # Convert caption to indices
        caption_ids = self.vocab.caption_to_indices(caption)
        captions_tensor = torch.tensor(caption_ids)

        return image, captions_tensor


class CollateFn:
    """
    Collate function to use in the DataLoader.

    Attributes:
        pad_idx (int): index of the padding token.
    """
    def __init__(self, pad_idx: int):
        """
        Initialize the CollateFn object.

        Args:
            pad_idx (int): index of the padding token.
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        Call the CollateFn object.

        Args:
            batch (list): list of tuples with the images and captions.

        Returns:
            tuple: tuple with the images and the padded captions.
        """
        images = [item[0].unsqueeze(0) for item in batch]
        captions = [item[1] for item in batch]

        # concat the images
        images = torch.cat(images, dim=0)

        # pad the captions
        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)

        return images, captions
