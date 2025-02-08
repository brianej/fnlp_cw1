import unittest

import jsonlines
from tokenizers import (
    NgramTokenizer,
    ReturnWordsTokenizer,
)
from models import (
    CountFeatureExtractor,
    MeanPoolingWordVectorFeatureExtractor,
    LogisticRegressionClassifier,
)
from utils import SentimentExample
from collections import Counter
import pickle
import numpy as np

with jsonlines.open("unittest_dataset.jsonl", "r") as reader:
    dataset = list(reader)

corpus = [datapoint["text"] for datapoint in dataset]

dummy_corpus = [
    "This movie was really bad, but bad in a fun way, so I loved it.",
    "The book series that this is based on is one of the best book series I have ever read, but this TV show is the worst TV show I have ever seen.",
]

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


class TokenizerTest(unittest.TestCase):
    def test_unigram_tokenize(self):
        unigram = NgramTokenizer(n=1)
        unigram.train(corpus)
        sample_text = (
            "This movie was really bad, but bad in a fun way, so I loved it."
        )
        tokenized = unigram.tokenize(sample_text)
        self.assertEqual(
            tokenized,
            [
                ("This",),
                ("movie",),
                ("was",),
                ("really",),
                ("bad",),
                (",",),
                ("but",),
                ("bad",),
                ("in",),
                ("a",),
                ("fun",),
                ("way",),
                (",",),
                ("so",),
                ("I",),
                ("loved",),
                ("it",),
                (".",),
            ],
        )

    def test_bigram_tokenize(self):
        ngram = NgramTokenizer(n=2)
        ngram.train(corpus)
        sample_text = (
            "This movie was really bad, but bad in a fun way, so I loved it."
        )
        tokenized = ngram.tokenize(sample_text)
        self.assertEqual(
            tokenized,
            [
                ("This", "movie"),
                ("movie", "was"),
                ("was", "really"),
                ("really", "bad"),
                ("bad", ","),
                (",", "but"),
                ("bad", "in"),
                ("in", "a"),
                ("a", "fun"),
                ("way", ","),
                (",", "so"),
                ("so", "I"),
                ("I", "loved"),
                ("loved", "it"),
                ("it", "."),
            ],
        )

    def test_trigram_tokenize(self):
        ngram = NgramTokenizer(n=3)
        ngram.train(corpus)
        sample_text = (
            "This movie was really bad, but bad in a fun way, so I loved it."
        )
        tokenized = ngram.tokenize(sample_text)
        self.assertEqual(
            tokenized,
            [
                ("This", "movie", "was"),
                ("movie", "was", "really"),
                ("was", "really", "bad"),
                ("bad", ",", "but"),
                ("in", "a", "fun"),
                (",", "so", "I"),
                ("I", "loved", "it"),
            ],
        )
        tokenized = ngram.tokenize(sample_text, return_token_ids=True)
        print(tokenized)

class CountFeatureExtractorTest(unittest.TestCase):
    def test_count_feature_extractor(self):
        unigram = NgramTokenizer(n=1)
        unigram.train(dummy_corpus)
        count_feature_extractor = CountFeatureExtractor(unigram)
        print(unigram)
        text = dummy_corpus[0]
        features = count_feature_extractor.extract_features(text)
        # you may have different order of features, but the count values should be the same
        true_counts = Counter(
            {
                4: 2,
                5: 2,
                0: 1,
                1: 1,
                2: 1,
                3: 1,
                6: 1,
                7: 1,
                8: 1,
                9: 1,
                10: 1,
                11: 1,
                12: 1,
                13: 1,
                14: 1,
                15: 1,
            }
        )
        true_count_values = Counter(true_counts.values())
        predicted_count_values = Counter(features.values())

        self.assertEqual(predicted_count_values, true_count_values)

class LogisticRegressionTest(unittest.TestCase):
    def test_predict(self):
        tokenizer = NgramTokenizer(n=1)
        tokenizer.token_to_id = {("hi",): 0, ("world",): 1, ("foo",): 2}
        tokenizer.id_to_token = {0: ("hi",), 1: ("world",), 2: ("foo",)}
        featurizer = CountFeatureExtractor(tokenizer)
        model = LogisticRegressionClassifier(featurizer)
        # set weights and bias
        model.weights = np.array([1, 2, 10])
        model.bias = 1
        prediction = model.predict("hi hi world")
        self.assertEqual(prediction, 1)
    
    def test_training_step(self):
        tokenizer = NgramTokenizer(n=1)
        tokenizer.token_to_id = {("hi",): 0, ("world",): 1, ("foo",): 2}
        tokenizer.id_to_token = {0: ("hi",), 1: ("world",), 2: ("foo",)}
        featurizer = CountFeatureExtractor(tokenizer)
        
        # first case, multiple examples and update weights
        examples = [SentimentExample(words="hi hi world", label=1), SentimentExample(words="foo bar", label=0)]
        learning_rate = 0.5
        model = LogisticRegressionClassifier(featurizer)
        model.weights = np.array([-2, 1, 2], dtype=np.float64)
        model.bias = -1
        
        prediction = model.predict(examples[0].words)
        self.assertEqual(prediction, 0)
        prediction = model.predict(examples[1].words)
        self.assertEqual(prediction, 1)

        model.training_step(examples, learning_rate)
        self.assertTrue(np.array_equal(model.weights, np.array([-1.5, 1.25, 1.75], dtype=np.float64)))
        self.assertEqual(model.bias, -1.0)
        
        # second case, one example and update bias
        examples = [SentimentExample(words="not foo foo", label=0)]
        learning_rate = 0.5
        model = LogisticRegressionClassifier(featurizer)
        model.weights = np.array([-2, 1, 2], dtype=np.float64)
        model.bias = -1
        
        prediction = model.predict(examples[0].words)
        self.assertEqual(prediction, 1)
        
        model.training_step(examples, learning_rate)
        self.assertTrue(np.array_equal(model.weights, np.array([-2, 1, 1], dtype=np.float64)))
        self.assertEqual(model.bias, -1.5)
        
        # third case, all correct predictions and no update
        examples = [SentimentExample(words="not foo foo", label=0), SentimentExample(words="hi world world", label=1)]
        learning_rate = 0.5
        model = LogisticRegressionClassifier(featurizer)
        model.weights = np.array([2, -1, -1], dtype=np.float64)
        model.bias = 0.5
        
        prediction = model.predict(examples[0].words)
        self.assertEqual(prediction, 0)
        prediction = model.predict(examples[1].words)
        self.assertEqual(prediction, 1)
        
        model.training_step(examples, learning_rate)
        self.assertTrue(np.array_equal(model.weights, np.array([2, -1, -1], dtype=np.float64)))
        self.assertEqual(model.bias, 0.5)
        
        
    def test_logistic_regression(self):
        tokenizer = NgramTokenizer(n=1)
        tokenizer.train(dummy_corpus)
        featurizer = CountFeatureExtractor(tokenizer)
        model = LogisticRegressionClassifier(featurizer)
        weights = model.weights
        bias = model.bias
        self.assertTrue(np.array_equal(weights, np.zeros(len(weights))))
        self.assertEqual(bias, 0)
        # run prediction before training (weight and bias should be 0)
        prediction = model.predict(dummy_corpus[0])
        # should be 1 because raw prediction is sigmoid(0)=0.5
        self.assertEqual(prediction, 1)

        # run one training step with learning rate 0.1
        # however, weights and bias should not change
        batch_exs = [SentimentExample(dummy_corpus[0], 1)]
        model.training_step(batch_exs, 1.0)
        self.assertTrue(np.array_equal(weights, np.zeros(len(weights))))
        self.assertEqual(bias, 0)

        # run another training step with learning rate 0.1
        # weights and bias should be updated
        batch_exs = [SentimentExample(dummy_corpus[1], 0)]
        model.training_step(batch_exs, 1.0)
        weights = model.weights
        bias = model.bias
        self.assertFalse(np.array_equal(weights, np.zeros(len(weights))))
        self.assertNotEqual(bias, 0)

        prediction = model.predict(dummy_corpus[1])
        self.assertEqual(prediction, 0)
        
class MeanPoolingWordVectorFeatureExtractorTest(unittest.TestCase):
    def test_mean_pooling_word_vector_feature_extractor(self):
        text = dummy_corpus[0]
        return_text_tokenizer = ReturnWordsTokenizer()
        mean_pooling_feature_extractor = MeanPoolingWordVectorFeatureExtractor(
            return_text_tokenizer
        )
        features = mean_pooling_feature_extractor.extract_features(text)
            
        with open("mean_pooling_feature_extractor_unittest_sol.pkl", "rb") as f:
            correct_features = pickle.load(f)
        
        # check each individual feature that it's close to the correct feature (some floating point error)
        for i in range(len(features)):
            self.assertTrue(np.isclose(features[i], correct_features[i]))
        
    def test_lr_with_mean_pooling(self):
        tokenizer = ReturnWordsTokenizer()
        tokenizer.train(dummy_corpus)
        featurizer = MeanPoolingWordVectorFeatureExtractor(tokenizer)
        model = LogisticRegressionClassifier(featurizer)
        weights = model.weights
        bias = model.bias
        self.assertTrue(np.array_equal(weights, np.zeros(len(weights))))
        self.assertEqual(bias, 0)
        
        batch_exs = [SentimentExample(dummy_corpus[1], 0)]
        model.training_step(batch_exs, 1.0)
        weights = model.weights
        bias = model.bias
        
        # load weights and bias in one dictionary
        with open("lr_meanpooling_weights_and_bias_unittest_sol.pkl", "rb") as f:
            weights_and_bias = pickle.load(f)
            
        correct_weights = weights_and_bias["weights"]
        correct_bias = weights_and_bias["bias"]
        
        self.assertTrue(np.allclose(weights, correct_weights))
        self.assertEqual(bias, correct_bias)

if __name__ == "__main__":
    unittest.main()
