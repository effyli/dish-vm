import transformers
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer

import os
import wandb
wandb.init(project="test-ner", entity="effyli")
os.environ["WANDB_API_KEY"]="8cf8498dc048e0d37a2de7bd0c3512b7ee7a7e2b"
os.environ["WANDB_ENTITY"]="Suchandra"
os.environ["WANDB_PROJECT"]="finetune_bert_base_ner"

task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "dslim/bert-base-NER"
batch_size = 16

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    # True prediction use label_list_model; true labels use label_list_data
    true_predictions = [
        [label_list_model[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list_data[l] for (p, l) in zip(prediction, label) if l != -100]
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
    # print("pre-processing")
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    label_all_tokens = True
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
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# second processing when model labels are different from dataset labels
def map_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

if __name__ == '__main__':
    seed = 2022
    transformers.set_seed(seed)

    datasets = load_dataset("conll2003")

    # this label list is different from the actual label list that model was trained on
    # we have to deal with the label mapping
    label_list_data = datasets['train'].features[f"{task}_tags"].feature.names
    label_dict_data = {k: v for k, v in enumerate(label_list_data)}
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list_data))
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, grouped_entities=True, ignore_subwords=True)

    label_dict_model = model.config.id2label
    label_dict_rev_model = model.config.label2id
    label_list_model = [v for k, v in sorted(label_dict_model.items(), key=lambda k:k[0])]
    # need matching
    label_map = {}
    if label_dict_data != label_dict_model:
        # we map the dataset label to model label
        for k, v in label_dict_data.items():
            k_model = label_dict_rev_model[v]
            label_map[k] = k_model
    else:
        for i in range(len(label_list_data)):
            label_map[i] = i

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    # when compute metrics, we use two label dict to convert label_ids

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
        push_to_hub=True,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")


    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    testing_example = "My name is Effy, and I am from Jiaozuo, Henan."
    print("after tokenization: ", tokenizer(testing_example))
    print("coming back to words: ", tokenizer.convert_ids_to_tokens(tokenizer(testing_example)["input_ids"]))


    # examples = load_dataset("conll2003", split="validation[0:16]")
    #
    # examples = examples.map(tokenize_and_align_labels, batched=True)
    # predictions, labels, _ = trainer.predict(examples)
    # predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list_model[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list_data[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)
    trainer.push_to_hub()

    wandb.finish()
