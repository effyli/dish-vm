import json
import argparse

from embedding_generator import sent_embedding_generator
from sentence_transformers import SentenceTransformer

def join_sentence(sentences):
    res = []
    for l in sentences:
        res.append(' '.join(l))
    return res


if __name__ == '__main__':
    sampled = True
    data_folder = "dataset/new_dish/"

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embedding_method',
                        choices=['token', 'sentence'], help='The method used for obtaining sentence representation')
    parser.add_argument('-b', '--batch_size', default=32, help='batch size for encoding using sentence bert')
    args = parser.parse_args()

    embedding_method = args.embedding_method
    batch_size = args.batch_size

    # we compare different writing styles
    # such as news and conversations
    datasets = ["conll_dish.json", "cerec_dish.json", "ontonotes_dish.json", "i2b2-06_dish.json",
                "GUM_dish.json", "AnEM_dish.json", "BTC_dish.json", "WNUT17_dish.json", "Wikigold_dish.json",
                "re3d_dish.json", "SEC_dish.json", "sciERC_dish.json"]
    names = ["conll", "cerec", "ontonotes", "i2b2", "GUM", "AnEM", "BTC", "WNUT17", "wikigold", "re3d", "sec", "sciERC"]
    print("Number of datasets: ", len(datasets))

    data_list = []
    # mentions_list = []
    for dataset in datasets:
        if sampled:
            with open(data_folder + dataset.split('.')[0] + '_sampled.json', 'r') as f:
                output = json.load(f)
                data_list.append(output['data'])
                # mentions_list.append(output['mentions'])
        else:
            with open(data_folder + dataset, 'r') as f:
                output = json.load(f)
                data_list.append(output['data'])
                # mentions_list.append(output['mentions'])

    if embedding_method == 'token':
        # average token representation for each sentence
        path = 'test_embedding_files/'
        for data, name in zip(data_list, names):
            embedding_list = []
            for sent in data[:10]:
                embedding_list.append(sent_embedding_generator(sent).cpu().numpy().tolist())
            with open(path + name +'_embeddings', 'w') as f:
                json.dump(embedding_list, f)

    elif embedding_method == 'sentence':
        # use sentence bert
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        path = 'sent_embedding_files/'

        for data, name in zip(data_list, names):
            data = join_sentence(data)
            # Sentences are encoded by calling model.encode()
            embeddings = model.encode(data, batch_size=batch_size)
            embeddings = embeddings.tolist()

            with open(path + name + '_embeddings', 'w') as f:
                json.dump(embeddings, f)
            print('Finished processing dataset: ', name)

