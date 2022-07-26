import os
import json
import logging
import argparse
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from seqeval.metrics import classification_report


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

def process_entity_type(mentions):
    # process entity_type to be LOC, PER, or ORG
    for mention in mentions:
        if 'O' in mention['entity_type']:
            mention['entity_type'] = 'ORG'
        elif 'P' in mention['entity_type']:
            mention['entity_type'] = 'PER'
        elif 'L' in mention['entity_type']:
            mention['entity_type'] = 'LOC'
        elif 'D' in mention['entity_type']:
            mention['entity_type'] = 'DIG'
    return mentions


def get_mention(token, mentions):
    # a function to return label such as B-PER, I-PER, O
    iob_token = 'O'
    doc_id, token = token
    for m in mentions:
        if doc_id == m['doc_id']:
            if token[1] in m['tokens_ids']:
                if token[1] == m['tokens_ids'][0]:
                    prefix = 'B'
                else:
                    prefix = 'I'
                iob_token = prefix + '-' + m['entity_type']
    return iob_token


def data_loader(data, labels):
    samples = []
    for sent, gold_labels in zip(data, labels):
        samples.append((sent, gold_labels))
    return samples


def tokens_annotations_alignment(sample, label, tokenzier):
    annotation = ['O']
    sample_lst = sample.split(' ')
    offset_mappings = tokenzier(sample_lst, return_offsets_mapping=True).data['offset_mapping']
    # skip bad samples
    if len(offset_mappings) != len(labels):
        return None
    for i, tokens in enumerate(offset_mappings):
        for token in tokens:
            if token != (0, 0):
                annotation.append(label[i])
    annotation.append('O')
    return annotation


def process_ner_res(ner_res, length):
    res = ['O' for _ in range(length)]
    for r in ner_res:
        res[r['index']] = r['entity']
    return res


if __name__ == '__main__':
    # This script is used for measure the performance and
    # calculate the performance difference for any pair of datasets
    logging.basicConfig(filename='performance_report',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', help='Path to input folder')
    parser.add_argument('-o', '--output_folder', help='Path to output folder for storing evaluation reports')
    args = parser.parse_args()

    logging.info("Running MMD testing for all datasets")

    # Set useful parameters
    data_folder = args.input_folder
    output_folder = args.output_folder

    datasets = ["conll_dish.json", "cerec_dish.json", "ontonotes_dish.json", "i2b2-06_dish.json",
                "GUM_dish.json", "AnEM_dish.json", "BTC_dish.json", "WNUT17_dish.json", "Wikigold_dish.json",
                "re3d_dish.json", "SEC_dish.json", "sciERC_dish.json"]
    # names following the same order
    names = ["conll", "cerec", "ontonotes", "i2b2-06", "GUM", "AnEM", "BTC", "WNUT17", "wikigold", "re3d", "SEC", "sciERC"]

    # datasets = ["BTC_dish.json", "WNUT17_dish.json", "Wikigold_dish.json",
    #             "re3d_dish.json", "SEC_dish.json", "sciERC_dish.json"]
    # # names following the same order
    # names = ["BTC", "WNUT17", "wikigold", "re3d", "SEC",
    #          "sciERC"]
    print("Number of datasets: ", len(datasets))
    results = {}
    for dataset, name in zip(datasets, names):
        # load data
        with open(data_folder + dataset, 'r') as f:
            output = json.load(f)
        data = get_sents(output['data'])
        # get labels
        with open(data_folder + '{}_labels_mapped_dish.json'.format(name), 'r') as f:
            labels = json.load(f)
        # loading model
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        nlp = pipeline("ner", model=model, tokenizer=tokenizer)

        all_labels = []
        all_preds = []
        # prepare test data
        for sample_tuple in data_loader(data, labels):
            sample, label = sample_tuple
            ner_results = nlp(sample)
            label_aligned = tokens_annotations_alignment(sample, label, tokenizer)
            if not label_aligned:
                continue
            pred = process_ner_res(ner_results, len(label_aligned))
            all_labels.append(label_aligned)
            all_preds.append(pred)
        print(name)
        logging.info(name+'\n')
        res = classification_report(all_labels, all_preds)
        print(res)
        logging.info(res)
        results[name] = res
    with open(output_folder + 'calssification_report.json', 'w') as f:
        json.dump(results, f)


