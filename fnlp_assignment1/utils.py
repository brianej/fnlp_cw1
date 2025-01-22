# utils.py
import jsonlines
from tokenizers import Tokenizer

from typing import List

class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (str): string of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """

    def __init__(self, words, label):
        self.words = words
        self.label = label

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()

def train_tokenizer(infile: str, max_corpus_size: int, tokenizer: Tokenizer):
    """
    Trains a tokenizer on the given file.
    """
    with jsonlines.open(infile, mode="r") as reader:
        data = list(reader)
    
    corpus = []
    for line in data:
        corpus.append(line["text"])
        if len(corpus) >= max_corpus_size:
            break
    
    tokenizer.train(corpus)

def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from the jsonlines file. Returns a list of SentimentExample objects.

    :param infile: file to read from
    :return: a list of SentimentExamples parsed from the file
    """
    with jsonlines.open(infile, mode="r") as reader:
        data = list(reader)
    
    exs = []
    for line in data:
        text = line["text"]
        label = line["label"]
        # add filtering for long sentences - 5000 is arbitrary but should work on DICE
        # if len(text.strip()) > 0 and len(text) < 5000:
        exs.append(SentimentExample(text, label))
    return exs


def read_blind_sst_examples(infile: str) -> List[List[str]]:
    """
    Reads the blind SST test set, which just consists of unlabeled texts
    :param infile: path to the file to read
    :return: list of tokenized texts (list of list of strings)
    """
    with jsonlines.open(infile, mode="r") as reader:
        data = list(reader)
    
    exs = []
    for line in data:
        text = line["text"]
        if len(text.strip()) > 0:
            exs.append(text.split(" "))
    return exs


def write_sentiment_examples(exs: List[SentimentExample], outfile: str):
    """
    Writes sentiment examples to an output file with one example per line, the predicted label followed by the example.
    Note that what gets written out is tokenized.
    :param exs: the list of SentimentExamples to write
    :param outfile: out path
    :return: None
    """
    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.label) + "\t" + " ".join([word for word in ex.words]) + "\n")
    o.close()
