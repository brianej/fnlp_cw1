# sentiment_classifier.py

import argparse
import sys
from tokenizers import (
    NgramTokenizer,
    Tokenizer,
    ReturnWordsTokenizer,
)
from models import train_model
from utils import train_tokenizer, read_sentiment_examples
import time
from typing import List
import jsonlines

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description="trainer.py")
    parser.add_argument(
        "--model", type=str, default="TRIVIAL", help="model to run (TRIVIAL, or LR)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="NGRAM",
        help="tokenizer to use (NONE, or NGRAM)",
    )
    parser.add_argument(
        "--feats",
        type=str,
        default="COUNTER",
        help="features to use (COUNTER, WV, CUSTOM)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.5, help="learning rate"
    )
    parser.add_argument(
        "--write_predictions",
        action="store_true",
        help="write predictions to a file",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--ngrams", type=int, default=2, help="ngrams to use (2)")
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/imdb_train.txt",
        help="path to train set (you should not need to modify)",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="data/imdb_dev.txt",
        help="path to dev set (you should not need to modify)",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/imdb_test.txt",
        help="path to test set (you should not need to modify)",
    )
    args = parser.parse_args()
    return args


def get_tokenizer(args):
    if args.tokenizer == "NONE":
        return ReturnWordsTokenizer()
    elif args.tokenizer == "NGRAM":
        return NgramTokenizer(n=args.ngrams)
    else:
        raise Tokenizer()


def evaluate(
    classifier,
    exs,
    write_to_file: bool = False,
    save_path: str = "test_predictions.jsonl",
):
    """
    Evaluates a given classifier on the given examples
    :param classifier: classifier to evaluate
    :param exs: the list of SentimentExamples to evaluate on
    :return: None (but prints output)
    """
    predictions = [classifier.predict(ex.words) for ex in exs]
    print_evaluation([ex.label for ex in exs], predictions)

    if write_to_file:
        with jsonlines.open(save_path, mode="w") as writer:
            for ex, prediction in zip(exs, predictions):
                writer.write(
                    {"gold": ex.label, "prediction": prediction, "words": ex.words}
                )


def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints evaluation statistics comparing golds and predictions, each of which is a sequence of 0/1 labels.
    Prints accuracy as well as precision/recall/F1 of the positive class, which can sometimes be informative if either
    the golds or predictions are highly biased.

    :param golds: gold labels
    :param predictions: pred labels
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception(
            "Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions))
        )
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print(
        f"Accuracy: {num_correct} / {num_total} = {float(num_correct) / num_total:0.4f}"
    )
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print(
        f"Precision: {num_pos_correct} / {num_pred} = {prec:0.4f}"
    )
    print(
        f"Recall: {num_pos_correct} / {num_gold} = {rec:0.4f}"
    )
    print(
        f"F1 (harmonic mean of precision and recall): {f1:0.4f}"
    )


if __name__ == "__main__":
    args = _parse_args()
    print(args)

    tokenizer = get_tokenizer(args)

    # set max corpus size to 1000 for training, otherwise it will can take too long
    train_tokenizer(args.train_path, 1000, tokenizer)

    # Load train, dev, and test exs and index the words.
    train_exs = read_sentiment_examples(args.train_path)
    dev_exs = read_sentiment_examples(args.dev_path)
    test_exs = read_sentiment_examples(args.test_path)

    print(
        repr(len(train_exs))
        + " / "
        + repr(len(dev_exs))
        + " / "
        + repr(len(test_exs))
        + " train/dev/test examples"
    )

    # Train and evaluate
    start_time = time.time()
    model = train_model(
        args,
        train_exs,
        dev_exs,
        tokenizer,
        args.learning_rate,
        args.batch_size,
        args.epochs,
    )
    print("=====Train Accuracy=====")
    evaluate(model, train_exs)
    print("=====Dev Accuracy=====")
    evaluate(model, dev_exs)
    print("Time for training and evaluation: %.2f seconds" % (time.time() - start_time))
    print("=====Test Accuracy=====")
    # modify save path with args, which feature and tokenizer are used
    feature_str = args.feats  # COUNTER, WV or CUSTOM
    tokenizer_str = f"NGRAM{args.ngrams}" if args.tokenizer == "NGRAM" else "NONE"
    hparam_str = f"lr{args.learning_rate}_bs{args.batch_size}_epochs{args.epochs}"
    save_path = (
        f"test_predictions_{feature_str}_{tokenizer_str}_{hparam_str}.jsonl"
        if args.write_predictions
        else None
    )
    evaluate(model, test_exs, write_to_file=args.write_predictions, save_path=save_path)
