import jsonlines
from datasets import load_dataset
import random
import os

seed_num = 10
random.seed(seed_num)

imdb = load_dataset("imdb")

train_dataset = imdb["train"]
# train_dataset = train_dataset.filter(lambda x, idx: idx < 10000, with_indices=True)
# imdb has no dev set, so we'll split the training set into 80% train and 20% dev
train_dataset, dev_dataset = train_dataset.train_test_split(test_size=0.2, seed=seed_num).values()

test_dataset = imdb["test"]

output_train_dataset_fname = "data/imdb_train.txt"
output_dev_dataset_fname = "data/imdb_dev.txt"
output_test_dataset_fname = "data/imdb_test.txt"

# create data folder if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

train_datapoints = [{"text": datapoint['text'], "label": datapoint['label']} for datapoint in train_dataset]
train_datapoints = train_datapoints[:10000]
dev_datapoints = [{"text": datapoint['text'], "label": datapoint['label']} for datapoint in dev_dataset]
dev_datapoints = dev_datapoints[:2000]
test_datapoints = [{"text": datapoint['text'], "label": datapoint['label']} for datapoint in test_dataset]

with jsonlines.open(output_train_dataset_fname, mode="w") as writer:
    writer.write_all(train_datapoints)

with jsonlines.open(output_dev_dataset_fname, mode="w") as writer:
    writer.write_all(dev_datapoints)

with jsonlines.open(output_test_dataset_fname, mode="w") as writer:
    writer.write_all(test_datapoints)

