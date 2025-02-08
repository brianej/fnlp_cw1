from typing import List
import gensim.downloader as api
from gensim.models import Word2Vec
from utils import read_sentiment_examples
import string

def train_word2vec_model(sentences: List[List[str]]) -> Word2Vec:
    """
    Train a word2vec model using gensim, should have vector size of 300, min_count of 1
    Should be ~1 line of code
    """
    model = Word2Vec(sentences= sentences, vector_size= 300, min_count=1)
    return model
    raise Exception("TODO: Implement this method")
    
def simple_text_to_words(text: str) -> List[str]:
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split(" ")
    
print("Loading pretrained word to vector model...")
# load the pretrained word to vector model
pretrained_word_to_vector_model = api.load("glove-twitter-25")

print("Loading training, test, and validation examples...")
# load the training, test, and validation examples
train_exs = read_sentiment_examples("data/imdb_train.txt")
test_exs = read_sentiment_examples("data/imdb_test.txt")
val_exs = read_sentiment_examples("data/imdb_dev.txt")
all_exs = train_exs + test_exs + val_exs

# convert the examples to words
# sentences = [convert_text_to_words(ex.words) for ex in all_exs]
sentences = [simple_text_to_words(ex.words) for ex in all_exs]


print("Training word2vec model...")
model = train_word2vec_model(sentences)

print("Getting most similar words...")

# get the most similar words
words_of_interest = ['angle', 'shot', 'realistic', 'computer']
for word in words_of_interest:
    our_most_similar = model.wv.most_similar(word, topn=5)
    our_most_similar_words = [x[0] for x in our_most_similar]
    pretrained_most_similar = pretrained_word_to_vector_model.most_similar(word, topn=5)
    pretrained_most_similar_words = [x[0] for x in pretrained_most_similar]
    print(f"{word}:")
    print(f"    Our model: {our_most_similar_words}")
    print(f"    Pretrained model: {pretrained_most_similar_words}")

