import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric, Dataset
import numpy as np
import argparse
import json
import pandas as pd

import os
import wandb
wandb.init(project="dish-bert-perf", entity="effyli")
os.environ["WANDB_API_KEY"]="8cf8498dc048e0d37a2de7bd0c3512b7ee7a7e2b"
os.environ["WANDB_ENTITY"]="Suchandra"
os.environ["WANDB_PROJECT"]="dish-bert-perf"

task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "bert-base-uncased"
batch_size = 16

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list_model[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_map[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def load_data_from_json(data_folder, name, mode):
    # create a pandas dataframe first then contruct hf dataset
    if mode == "train":
        with open(data_folder + "{}_dish_train.json".format(name), 'r') as f:
            train_data = json.load(f)
        with open(data_folder + "{}_dish_train_labels.json".format(name), 'r') as f:
            train_labels = json.load(f)
        label_list = list(set(t for e in train_labels for t in e))
        train_df = pd.DataFrame(list(zip(train_data, train_labels)), columns=['tokens', 'ner_tags'])
        train_dataset = Dataset.from_pandas(train_df)
    elif mode == "test":
        with open(data_folder + "{}_dish_test.json".format(name), 'r') as f:
            test_data = json.load(f)
        with open(data_folder + "{}_dish_test_labels.json".format(name), 'r') as f:
            test_labels = json.load(f)

        test_df = pd.DataFrame(list(zip(test_data, test_labels)), columns=['data', 'labels'])
        test_dataset = Dataset.from_pandas(test_df)
    if mode == "train":
        return train_dataset, label_list
    elif mode == "test":
        return test_dataset


if __name__ == '__main__':
    seed = 2022
    transformers.set_seed(seed)

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', help='Path to input folder')
    parser.add_argument('-e', '--num_epoch', type=int, help='Number of epochs used to fine-tune the model')
    parser.add_argument('-s', '--skip_training', help='Command to skip training and evaluating only',
                        default=False, action='store_true')
    args = parser.parse_args()

    # Set useful parameters
    data_folder = args.input_folder
    epoch = args.num_epoch

    datasets = ["conll_dish.json", "ontonotes_dish.json", "i2b2-06_dish.json",
                "GUM_dish.json", "AnEM_dish.json", "BTC_dish.json", "WNUT17_dish.json", "Wikigold_dish.json",
                "re3d_dish.json", "SEC_dish.json", "sciERC_dish.json"]
    # names following the same order
    names = ["conll", "ontonotes", "i2b2-06", "GUM", "AnEM", "BTC", "WNUT17", "wikigold", "re3d", "SEC", "sciERC"]

    print("Number of datasets: ", len(datasets))
    for s_name in names:
        # we get the source dataset, and fine-tune on the source training set, we then evaluate on all the test sets
        print('Processing source dataset {}'.format(s_name))
        train_dataset, label_list = load_data_from_json(data_folder, s_name, "train")
        label_map = {v: k for k, v in enumerate(label_list)}
        print(label_list)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
        tokenized_train_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

        model_name = model_checkpoint.split("/")[-1]
        args = TrainingArguments(
            f"{model_name}-finetuned-{task}",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            report_to="wandb",
            weight_decay=0.01,
            push_to_hub=False,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)
        metric = load_metric("seqeval")

        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_train_datasets,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()

        for t_name in names:
            print("Processing dataset {} and {}".format(s_name, t_name))
            test_dataset = load_dataset(data_folder, t_name, 'test')
            trainer.evaluate(test_dataset)

    wandb.finish()







