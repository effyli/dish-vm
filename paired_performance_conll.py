import os
import json
import torch
import random
import logging
import argparse
import itertools
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

from transformers import pipeline
from transformers import AdamW, AutoTokenizer, AutoModelForTokenClassification

from seqeval.metrics import classification_report
from dataset import CustomizedDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_f1(result_str, category=None):
    if category:
        if category not in result_str:
            return None
        else:

            return result_str.split(category)[1].split('      ')[3]
    else:
        return result_str.split('weighted avg')[1].split('      ')[3]


def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def get_sents(tokens):
    res = []
    # concatenate a list of tokens to sentences
    for sent_t in tokens:
        sent = ''
        for t in sent_t:
            sent += t
            sent += ' '
        res.append(sent.strip())
    return res


def evaluate(val_dataset, device, model, id2tag, categories=None):
    # evaluation
    model.eval()
    print('evaluating...')
    all_labels = []
    all_preds = []

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        all_labels.append(labels.squeeze(0).tolist())
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        all_preds.append(pred.squeeze(0).tolist())
    # model.train()
    labels_truncated = []
    preds_truncated = []
    # dealing with the ignored preds
    for labels, preds in zip(all_labels, all_preds):
        if len(labels) != len(preds):
            print('bad example')
            continue
        arr_labels = np.array(labels)
        preds_truncated.append(np.array(preds)[arr_labels != -100].tolist())
        labels_truncated.append(arr_labels[arr_labels != -100].tolist())

    val_labels = [[id2tag[tid] for tid in sent] for sent in labels_truncated]
    all_preds = [[id2tag[tid] for tid in sent] for sent in preds_truncated]

    result_str = classification_report(val_labels, all_preds)
    if categories:
        for key in categories.keys():
            categories[key].append(get_f1(result_str, key))
        return get_f1(result_str), result_str, categories
    else:
        return get_f1(result_str), result_str


def get_data_labels(data_folder, p_names):
    p_data = []
    p_labels = []
    # load data
    for name in p_names:
        with open(data_folder + '{}_dish.json'.format(name), 'r') as f:
            output = json.load(f)
        data = get_sents(output['data'])
        # get labels
        with open(data_folder + '{}_labels_mapped_dish.json'.format(name), 'r') as f:
            labels = json.load(f)
        p_data.append(data)
        p_labels.append(labels)
    return p_data, p_labels


def load_data(data_folder, name):
    with open(data_folder + '{}_dish_train.json'.format(name), 'r') as f:
        train_data = json.load(f)
    with open(data_folder + '{}_dish_test.json'.format(name), 'r') as f:
        test_data = json.load(f)
    with open(data_folder + '{}_dish_train_labels.json'.format(name), 'r') as f:
        train_labels = json.load(f)
    with open(data_folder + '{}_dish_test_labels.json'.format(name), 'r') as f:
        test_labels = json.load(f)
    return train_data, train_labels, test_data, test_labels


def get_tag2id(labels):
    unique_tags = set(tag for doc in labels for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    return tag2id, id2tag


def trainer(train_data, train_labels, test_data, test_labels, weighted_f1s_categories,
            tag_id_maps, num_epoch=10, batch_size=32, model_name="dslim/bert-base-NER", evaluate_only=False):
    batch_size = batch_size
    num_epoch = num_epoch
    tag2id, id2tag = tag_id_maps
    # we fine-tune the model on train data and evaluate on test set
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(tag2id),
                                                            ignore_mismatched_sizes=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    id2tag = model.config.id2label
    tag2id = {tag: id for id, tag in id2tag.items()}

    train_encodings = tokenizer(train_data, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True)
    test_encodings = tokenizer(test_data, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                              truncation=True)

    train_labels = encode_tags(train_labels, train_encodings, tag2id)
    test_labels = encode_tags(test_labels, test_encodings, tag2id)

    train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
    test_encodings.pop("offset_mapping")
    train_dataset = CustomizedDataset(train_encodings, train_labels)
    test_dataset = CustomizedDataset(test_encodings, test_labels)
    print("test size:", len(test_dataset))
    logging.info('test size: ' + str(len(test_dataset)))

    if evaluate_only:
        model.eval()
        # # for debugging:
        # res = []
        # while True:
        #     outputs = evaluate(test_dataset, device, model, id2tag, weighted_f1s_categories)
        #     res.append(outputs)
        weighted_f1, result_str, weighted_f1s_categories = evaluate(test_dataset, device, model, id2tag, weighted_f1s_categories)
        print(result_str)
        return weighted_f1, result_str, weighted_f1s_categories

    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        model.train()

        optim = AdamW(model.parameters(), lr=5e-5)

        for epoch in range(num_epoch):
            print('Epoch ', epoch)
            epoch_loss = 0
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                # print('Loss: ', loss.item())
                epoch_loss += loss
                loss.backward()
                optim.step()
            print('Epoch loss: ', epoch_loss)

        weighted_f1, result_str, weighted_f1s_categories = evaluate(test_dataset, device, model, id2tag, weighted_f1s_categories)
        # TODO: dump all_reports to log file
        print(result_str)
        logging.info(result_str)
        return weighted_f1, result_str, weighted_f1s_categories


if __name__ == '__main__':
    DEBUG = False
    # This script is used for measure the performance and
    # calculate the performance difference for any pair of datasets
    logging.basicConfig(filename='performance_on_conll',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', help='Path to input folder')
    parser.add_argument('-o', '--output_folder', help='Path to output folder for storing evaluation reports')
    args = parser.parse_args()

    seed = 2022
    set_seed(seed)

    logging.info("Testing Bert for conll dataset")

    # Set useful parameters
    data_folder = args.input_folder
    output_folder = args.output_folder

    datasets = ["conll_dish.json"]
    # names following the same order
    names = ["conll"]
    print("Number of datasets: ", len(datasets))

    results = {}
    for t_name in names:
        s_train_data, s_train_labels, t_test_data, t_test_labels = load_data(data_folder, t_name)
        tag2id, id2tag = get_tag2id(s_train_labels + t_test_labels)
        weighted_f1_categories = {'ORG': [], 'PER': [], 'DIG': [], 'LOC': []}
        if DEBUG:
            s_train_data, s_train_labels, t_test_data, t_test_labels = s_train_data[:10], \
                                                                       s_train_labels[:10], t_test_data[:10], t_test_labels[:10]
        epoch = 1 if DEBUG else 10
        weighted_f1, result_str, weighted_f1s_categories = trainer(s_train_data, s_train_labels, t_test_data,
                                                                   t_test_labels, weighted_f1_categories,
                                                                   (tag2id, id2tag),
                                                                   num_epoch=epoch, evaluate_only=True)
        name_key = t_name
        results[name_key] = (weighted_f1, result_str, weighted_f1s_categories)

    with open(output_folder + 'perf_measure_conll', 'a+') as f:
        json.dump(results, f)






