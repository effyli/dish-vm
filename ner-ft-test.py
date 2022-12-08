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
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
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
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == '__main__':
    datasets_train = load_dataset("conll2003", split="train")
    datasets_val = load_dataset("conll2003", split="validation")
    datasets = {"train": datasets_train, "val": datasets_val}

    # this label list is different from the actual label list that model was trained on
    label_list_data = datasets_train.features[f"{task}_tags"].feature.names
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    tokenized_datasets_train = datasets_train.map(tokenize_and_align_labels, batched=True)
    tokenized_datasets_val = datasets_val.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list_data))
    label_dict = model.config.id2label
    # sort dictionary by keys
    label_list = [i[1] for i in sorted(label_dict.items(), key=lambda x:x[0])]

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
        train_dataset=datasets_train,
        eval_dataset=datasets_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # trainer.train()
    # trainer.evaluate()

    predictions, labels, _ = trainer.predict(tokenized_datasets_val)
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list_data[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)
    # trainer.push_to_hub()

    wandb.finish()
