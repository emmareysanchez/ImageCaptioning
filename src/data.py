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
            freq_threshold (int): The frequency threshold for including a word in the vocabulary.
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
    def tokenizer(text: str, extra_tokens: bool=False) -> list:
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
                if frequencies[word] == self.freq_threshold:
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
        """
        tokens = [self.idx2word.get(index, "<UNK>") for index in indices]
        return self.untokenizer(tokens)


class Flickr8kDataset(Dataset):
    def __init__(self, captions_path: str, images_path: str, transform=None, vocab=None):
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
        return len(self.image_names)

    def __getitem__(self, index):

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
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        captions = [item[1] for item in batch]

        # concat the images
        images = torch.cat(images, dim=0)

        # pad the captions
        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)

        return images, captions