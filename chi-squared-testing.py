import json
import itertools
from scipy.stats import chisquare, ks_2samp
from gensim.corpora import Dictionary

from utils import WordNotExistingError

import logging


def data_counts_by_names(names2compare, vocabs_to_compare, i_dict=None, j_dict=None):
    # i_dict can be a list of intersectional names or None
    # if i_dict is none, we then use all names in each vocab
    counts = []
    if i_dict is None and j_dict is None:
        vocabs_names = [list(v.values()) for v in vocabs_to_compare]
        for vocab,  vocab_names in zip(vocabs_to_compare, vocabs_names):
            count = []
            for n in vocab_names:
                if n not in vocab.values():
                    print(n)
                    raise WordNotExistingError
                count.append(vocab.cfs[vocab.token2id[n]])
            counts.append(count)
    elif i_dict != None:
        for vocab in vocabs_to_compare:
            count = []
            for n in i_dict:
                if n not in vocab.values():
                    print(n)
                    raise WordNotExistingError
                count.append(vocab.cfs[vocab.token2id[n]])
            counts.append(count)
    elif j_dict != None:
        for vocab in vocabs_to_compare:
            count = []
            for n in j_dict:
                if n not in vocab.values():
                    count.append(0)
                else:
                    count.append(vocab.cfs[vocab.token2id[n]])
            counts.append(count)
    return counts


def get_intersect_or_joined_vocabs(vocabs, method='joined'):
    joined_dict_names = [list(v.values()) for v in vocabs]
    u_dict_names = list(set().union(*joined_dict_names))
    i_dict_names = list(set.intersection(*map(set,joined_dict_names)))
    if method == 'intersect':
        return i_dict_names
    elif method == 'joined':
        return u_dict_names


def run_tests(names, vocabs, v_method):
    print('Vocabulary method ', v_method)
    logging.info('Vocabulary method ' + v_method)
    significance_level = 0.05
    num_tests = len(list(itertools.combinations_with_replacement(names, 2)))
    print("Number of tests that are carried out: ", num_tests)
    logging.info("Number of tests that are carried out: " + str(num_tests))

    for p_names in itertools.combinations_with_replacement(names, 2):
        print(p_names)
        logging.info('{}, {}'.format(p_names[0], p_names[1]))
        p_vocabs = [vocabs[names.index(n)] for n in p_names]
        proc_vocab = get_intersect_or_joined_vocabs(p_vocabs, method=v_method)
        if v_method == 'intersect':
            counts = data_counts_by_names(list(p_names), p_vocabs, i_dict=proc_vocab)
        elif v_method == 'joined':
            counts = data_counts_by_names(list(p_names), p_vocabs, j_dict=proc_vocab)
        sums = [sum(c) for c in counts]
        p_counts = []

        for c, s in zip(counts, sums):
            p_counts.append([(wc / s * 100) for wc in c])

        data1 = p_counts[0]
        data2 = p_counts[1]
        # ksres = ks_2samp(data1, data2, mode="asymp")
        chisq, p_val = chisquare(data1, data2)
        # perform Kolmogorov-Smirnov test
        print(chisq)
        # logging.info(' '.join(str(res) for res in list(ksres)))
        logging.info('chisquare result: {}, p_value: {}'.format(chisq, p_val))
        if p_val < significance_level:
            print("From different distribution")
            logging.info("From different distribution")
        else:
            print("From the same distributions")
            logging.info("From the same distributions")

    print('--------------------------------------------------')
    logging.info('--------------------------------------------------')


if __name__ == '__main__':

    logging.basicConfig(filename='chisqure-tests',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    logging.info("Running chisquare testing for all datasets")

    data_folder = "dataset/dish/"

    # we compare different writing styles
    # such as news and conversations
    datasets = ["conll_dish.json", "cerec_dish.json", "ontonotes_dish.json", "i2b2-06_dish.json",
                "GUM_dish.json", "AnEM_dish.json", "BTC_dish.json", "WNUT17_dish.json", "Wikigold_dish.json",
                "re3d_dish.json", "SEC_dish.json", "sciERC_dish.json"]
    names = ["conll", "cerec", "ontonotes", "i2b2", "GUM", "AnEM", "BTC", "WNUT17", "wikigold", "re3d", "sec", "sciERC"]
    print("Number of datasets: ", len(datasets))
    logging.info('number of datasets: '+str(len(datasets)))

    data_list = []
    mentions_list = []
    for dataset in datasets:
        with open(data_folder + dataset, 'r') as f:
            output = json.load(f)
            data_list.append(output['data'])
            mentions_list.append(output['mentions'])

    vocabs = []

    for name, data in zip(names, data_list):
        vocab = Dictionary(data)
        vocabs.append(Dictionary(data))

    run_tests(names, vocabs, v_method='intersect')
    # run_tests(names, vocabs, v_method='joined')


