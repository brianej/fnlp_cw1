# FNLP_Assignment1

## Setup
We recommend using a virtual environment to run the code. We used `conda create -n fnlp python=3.10 --file ./requirements.txt` to create our environment.
```
pip install -r requirements.txt
python -m gensim.downloader --download glove-twitter-25
```

## Setup data
```
python setup_data.py
```

## Train LR model with unigram and bigram tokenizers
```
python run_classifier.py --model LR --tokenizer NGRAM --ngrams 1 --feats COUNTER --learning_rate 0.5 --batch_size 2 --epochs 2
python run_classifier.py --model LR --tokenizer NGRAM --ngrams 2 --feats COUNTER --learning_rate 0.5 --batch_size 2 --epochs 2
python run_classifier.py --model LR --tokenizer NGRAM --ngrams 1 --feats CUSTOM --learning_rate 0.5 --batch_size 2 --epochs 2
python run_classifier.py --model LR --tokenizer NGRAM --ngrams 2 --feats CUSTOM --learning_rate 0.5 --batch_size 2 --epochs 2
```

Tune the hyperparameters learning rate, batch size, and epochs to improve performance (hyperparameters will be likely different for unigram and bigram tokenizers).

## Train LR model with word vectors

Note: this may take a while (5x longer) to run as getting the word vectors takes longer than your usual feature extraction.
```
python run_classifier.py --model LR --tokenizer NONE --feats WV --learning_rate 0.5 --batch_size 2 --epochs 2
```

## Run unit tests
```
python unit_tests.py
```

If you want to run a subset of the tests, you can run:
```
python unit_tests.py CLASS
```
where CLASS is the name of the class you want to test (LogisticRegressionTest, FeatureExtractorTest, TokenizerTest)

## Test tokenizers

In addition to the unit tests, you can also run the tokenizers.py file to inspect the tokenization results of your tokenizers. (Note: this also provides a way to answer the tokenization question from the assignment.)

```
python tokenizers.py
```

# Tasks

### Tokenizers

`tokenizers.py`: Implement the `NgramTokenizer`.

`NgramTokenizer`: Implement the `train, tokenize, and __len__` methods. This tokenizer should 'learn' (in `train()`) a corpus of n-grams and assign a unique ID to each n-gram. Tokenize will then convert a text into a list of n-grams and return the list of ids (in `tokenize()`).

### Feature Extractors

`models.py`: Implement the two subclasses of `FeatureExtractor` (`CountFeatureExtractor`, `CustomFeatureExtractor`,`MeanPoolingWordVectorFeatureExtractor`).

`CountFeatureExtractor`: Implement the `extract_features` method. This feature extractor will count the number of times each feature (in this case, a token) appears in a text.

`CustomFeatureExtractor`: Implement the `extract_features` method. This feature extractor will extract custom features (of your own design).

`MeanPoolingWordVectorFeatureExtractor`: Implement the `extract_features` method. This feature extractor will average the word vectors of all words in a text. You will get these word vectors from `spacy`; look into using `self.nlp.vocab.vectors`.

### Logistic Regression

`models.py`: Implement the `LogisticRegressionClassifier` subclass of `SentimentClassifier`.

`LogisticRegressionClassifier`: Implement the `predict, training_step, get_top_features, get_bottom_features` methods.

* `predict`: This method will use the model to predict the sentiment of a text.
* `training_step`: This method will update the weights and bias of the model from a batch of examples.
* `get_top_features`: This method will return the names of the top `num_features` features by weight (features most associated with positive sentiment).
* `get_bottom_features`: This method will return the names of the bottom `num_features` features by weight (features most associated with negative sentiment).

### Similarity

`similarity.py`: Implement the `train_word2vec_model` function. This function will train a word2vec model using gensim. Loading and training the models can take a couple minutes.

Run with:
```
python similarity.py
```
