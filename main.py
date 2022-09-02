import argparse
import numpy as np

import logging

from shift_tester import *
import itertools
import json


def load_embeddings_from_file(p_names, emb_data_dir):
    a, b = p_names
    with open(emb_data_dir + a + '_embeddings', 'r') as f:
        res_a = json.load(f)
    with open(emb_data_dir + b + '_embeddings', 'r') as f:
        res_b = json.load(f)
    return np.asarray(res_a), np.asarray(res_b)


def main():
    logging.basicConfig(filename='mmd-pvals',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logging.info("Running MMD testing for all datasets")

    logger = logging.getLogger('mmd-pvals')

    data_folder = 'dataset/'
    datasets = ["conll_dish.json", "cerec_dish.json", "ontonotes_dish.json", "i2b2-06_dish.json",
                "GUM_dish.json", "AnEM_dish.json", "BTC_dish.json", "WNUT17_dish.json", "Wikigold_dish.json",
                "re3d_dish.json", "SEC_dish.json", "sciERC_dish.json"]
    names = ["conll", "cerec", "ontonotes", "i2b2", "GUM", "AnEM", "BTC", "WNUT17", "wikigold", "re3d", "sec", "sciERC"]
    print("Number of datasets: ", len(datasets))
    emb_data_dir = 'sent_embedding_files/new_dish/'
    test_type = 'Multiv'
    test_dim = TestDimensionality.Multi
    mt = MultidimensionalTest.MMD

    # work on mmd for now
    shift_tester = ShiftTester(dim=test_dim, mt=mt)
    p_vals = []
    res_to_save = {}
    # load embeddings to test
    for p_names in itertools.combinations_with_replacement(names, 2):
        print(p_names)
        emb_a, emb_b = load_embeddings_from_file(p_names, emb_data_dir)
        p_val = shift_tester.test_shift(emb_a[:50], emb_b[:50])
        print('MMD distance: ', p_val[0])
        print('P value: ', p_val)
        p_vals.append(p_val)
        p_val_to_save = [p_val[0], p_val[1]]
        logging.info('mmd: ' + str(p_val))
        logging.info('p value: ' + str(p_val[0]))
        res_to_save[' and '.join(p_names)] = p_val_to_save
    with open('pvals_res', 'w') as f:
        json.dump(res_to_save, f)


if __name__ == '__main__':
    main()
