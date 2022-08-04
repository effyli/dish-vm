import os
import json
import torch
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
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        all_labels.append(labels.squeeze(0).tolist())
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        all_preds.append(pred.squeeze(0).tolist())

    model.train()
    labels_truncated = []
    preds_truncated = []
    # dealing with the ignored preds
    for labels, preds in zip(all_labels, all_preds):
        if len(labels) != len(preds):
            print('bad example')
            logging.info('bad example')
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


def dataset_loader(data, tokenizer, labels, tag2id):
    encodings = tokenizer(data, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True)
    labels = encode_tags(labels, encodings, tag2id)
    encodings.pop('offset_mapping')
    dataset = CustomizedDataset(encodings, labels)
    return dataset


def train(train_dataset, batch_size, model, num_epoch, device):
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
    return model


def trainer(tag_id_maps, train_data=None, train_labels=None, test_data=None, test_labels=None, num_epoch=10,
            batch_size=32, weighted_f1s_categories=None, model_name="dslim/bert-base-NER", mode='train', model=None):
    batch_size = batch_size
    num_epoch = num_epoch
    tag2id, id2tag = tag_id_maps
    # we fine-tune the model on train data and evaluate on test set
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if mode == 'train':
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(tag2id),
                                                                ignore_mismatched_sizes=True)
    else:
        model = model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    tag2id_model = model.config.label2id
    for tag, id in tag2id.items():
        if tag not in tag2id_model.keys():
            tag2id_model[tag] = len(tag2id_model)
    tag2id = tag2id_model
    id2tag = {id: tag for tag, id in tag2id.items()}

    if mode == 'evaluate':
        if test_data == None or test_labels == None:
            print('Need to provide test data')
            logging.info('Need to provide test data')
            raise ValueError
        test_dataset = dataset_loader(test_data, tokenizer, test_labels, tag2id)
        print("test size:", len(test_dataset))
        logging.info('test size: ' + str(len(test_dataset)))
        model.eval()
        weighted_f1, result_str, weighted_f1s_categories = evaluate(test_dataset, device, model, id2tag, weighted_f1s_categories)
        print(result_str)
        logging.info(result_str)
        return weighted_f1, result_str, weighted_f1s_categories

    elif mode == 'train':
        model.train()
        train_dataset = dataset_loader(train_data, tokenizer, train_labels, tag2id)
        print("train size:", len(train_dataset))
        logging.info('train size: ' + str(len(train_dataset)))
        model = train(train_dataset, batch_size, model, num_epoch, device=device)
        return model


if __name__ == '__main__':
    DEBUG = False
    # This script is used for measure the performance and
    # calculate the performance difference for any pair of datasets
    logging.basicConfig(filename='paired_performance_report',
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
    torch.manual_seed(seed)
    np.random.seed(0)


    logging.info("Testing Bert for all paired datasets")

    # Set useful parameters
    data_folder = args.input_folder
    output_folder = args.output_folder

    datasets = ["conll_dish.json", "cerec_dish.json", "ontonotes_dish.json", "i2b2-06_dish.json",
                "GUM_dish.json", "AnEM_dish.json", "BTC_dish.json", "WNUT17_dish.json", "Wikigold_dish.json",
                "re3d_dish.json", "SEC_dish.json", "sciERC_dish.json"]
    # names following the same order
    names = ["conll", "cerec", "ontonotes", "i2b2-06", "GUM", "AnEM", "BTC", "WNUT17", "wikigold", "re3d", "SEC", "sciERC"]
    print("Number of datasets: ", len(datasets))

    results = {}
    epoch = 10
    # all_reports = []
    # we do a two-sided paring experiments, so for some distance, we have two data points for the performance difference
    for s_name in names:
        # we get the source dataset, and fine-tune on the source training set, we then evaluate on all the test sets
        print('Processing source dataset {}'.format(s_name))
        logging.info('Processing source dataset {}'.format(s_name))
        logging.info('loading data')
        s_train_data, s_train_labels, _, _ = load_data(data_folder, s_name)
        tag2id, id2tag = get_tag2id(s_train_labels)
        if DEBUG:
            s_train_data, s_train_labels = s_train_data[10:20], s_train_labels[10:20]
            epoch = 1
        model = trainer((tag2id, id2tag), train_data=s_train_data, train_labels=s_train_labels, num_epoch=epoch, mode='train')
        for t_name in names:
            # getting the target dataset that we wants to evaluate on
            # we keep the source dataset fixed this way, fine-tune on the source dataset and evaluate on the rest of the datasets
            print("Processing dataset {} and {}".format(s_name, t_name))
            logging.info('Paired datasets names: {} and {}'.format(s_name, t_name))
            print('loading data')
            weighted_f1_categories = {'ORG': [], 'PER': [], 'DIG': [], 'LOC': []}
            # we then evaluate the fine-tuned model on the test set of the target dataset
            _, _, t_test_data, t_test_labels = load_data(data_folder, t_name)
            tag2id, id2tag = get_tag2id(t_test_labels)
            if DEBUG:
                t_test_data, t_test_labels = t_test_data[10:20], t_test_labels[10:20]
            weighted_f1, result_str, weighted_f1s_categories = trainer((tag2id, id2tag), test_data=t_test_data,
                                                                       test_labels=t_test_labels, weighted_f1s_categories=weighted_f1_categories, mode='evaluate', model=model)

            name_key = s_name + ' ' + t_name
            results[name_key] = (weighted_f1, result_str, weighted_f1s_categories)

    with open(output_folder + 'paird_perf_measure', 'a+') as f:
        json.dump(results, f)





