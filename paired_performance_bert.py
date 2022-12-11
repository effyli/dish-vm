import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric, Dataset
import numpy as np
import argparse
import json
import pandas as pd
from transformers.trainer_pt_utils import nested_concat, nested_detach, nested_numpify

import os
import wandb
wandb.init(project="dish-bert-perf", entity="effyli")
os.environ["WANDB_API_KEY"]="8cf8498dc048e0d37a2de7bd0c3512b7ee7a7e2b"
os.environ["WANDB_ENTITY"]="Suchandra"
os.environ["WANDB_PROJECT"]="dish-bert-perf"

task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "bert-base-uncased"
batch_size = 16
label_map = {}


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

def tokenize_and_align_labels_for_prediction(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    return tokenized_inputs

def load_data_from_json(data_folder, name, mode):
    # create a pandas dataframe first then contruct hf dataset
    if mode == "train":
        with open(data_folder + "{}_dish_train.json".format(name), 'r') as f:
            train_data = json.load(f)
        with open(data_folder + "{}_dish_train_labels.json".format(name), 'r') as f:
            train_labels = json.load(f)
        with open(data_folder + "{}_dish_test.json".format(name), 'r') as f:
            test_data = json.load(f)
        with open(data_folder + "{}_dish_test_labels.json".format(name), 'r') as f:
            test_labels = json.load(f)
        label_list = list(set(t for e in train_labels for t in e))
        train_df = pd.DataFrame(list(zip(train_data, train_labels)), columns=['tokens', 'ner_tags'])
        train_dataset = Dataset.from_pandas(train_df)
        test_df = pd.DataFrame(list(zip(test_data, test_labels)), columns=['tokens', 'ner_tags'])
        test_dataset = Dataset.from_pandas(test_df)
    elif mode == "test":
        with open(data_folder + "{}_dish_test.json".format(name), 'r') as f:
            test_data = json.load(f)
        with open(data_folder + "{}_dish_test_labels.json".format(name), 'r') as f:
            test_labels = json.load(f)
        test_label_list = list(set(t for e in test_labels for t in e))

        test_df = pd.DataFrame(list(zip(test_data, test_labels)), columns=['tokens', 'ner_tags'])
        test_dataset = Dataset.from_pandas(test_df)
    if mode == "train":
        return train_dataset, test_dataset, label_list
    elif mode == "test":
        return test_dataset, test_label_list


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

    # names following the same order
    names = ["conll", "ontonotes", "i2b2-06", "GUM", "AnEM", "BTC", "WNUT17", "wikigold", "re3d", "SEC", "sciERC"]

    print("Number of datasets: ", len(names))
    for s_name in names:
        # we get the source dataset, and fine-tune on the source training set, we then evaluate on all the test sets
        print('Processing source dataset {}'.format(s_name))
        train_dataset, test_dataset, label_list = load_data_from_json(data_folder, s_name, "train")
        label_map = {v: k for k, v in enumerate(label_list)}
        print(label_list)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

        tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
        tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

        model_name = model_checkpoint.split("/")[-1]
        args = TrainingArguments(
            f"{model_name}-finetuned-{task}",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch,
            report_to="wandb",
            weight_decay=0.01,
            push_to_hub=False,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)
        metric = load_metric("seqeval")


        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
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

        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_test_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()

        for t_name in names:
            tmp_label_map = label_map.copy()
            tmp_label_list = label_list.copy()
            print("Processing dataset {} and {}".format(s_name, t_name))
            test_dataset, test_label_list = load_data_from_json(data_folder, t_name, 'test')
            for label in test_label_list:
                if label not in label_list:
                    key = len(label_map)
                    label_map[label] = key
            print(label_map)

            tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels_for_prediction, batched=True)
            reverse_label_map = {v: k for k, v in label_map.items()}
            label_list = [v for k, v in sorted(reverse_label_map.items(), key=lambda k:k[0])]
            trainer.model.config.label2id = label_map
            trainer.model.config.id2label = reverse_label_map
            # trainer.model.num_labels = len(label_map)
            predictions, _, _ = trainer.predict(tokenized_test_dataset)
            # predictions = np.argmax(predictions, axis=2)
            tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
            # manually mimic the way bert prepare labels for evaluating
            labels = tokenized_test_dataset["labels"]
            labels_host = None
            for step, inputs in enumerate(trainer.get_eval_dataloader(tokenized_test_dataset)):
                inputs = trainer._prepare_inputs(inputs)
                labels = inputs["labels"]
                labels = nested_detach(labels)
                labels = trainer._pad_across_processes(labels)
                labels = trainer._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            labels = nested_numpify(labels_host)
            compute_metrics((predictions, labels))
            # trainer.evaluate(tokenized_test_dataset)
            label_map = tmp_label_map.copy()
            label_list = tmp_label_list.copy()
            print()


    wandb.finish()







