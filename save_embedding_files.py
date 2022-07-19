import json

from embedding_generator import sent_embedding_generator



if __name__ == '__main__':
    data_folder = "dataset/dish/"

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
        with open(data_folder + dataset, 'r') as f:
            output = json.load(f)
            data_list.append(output['data'])
            # mentions_list.append(output['mentions'])

    path = 'embedding_files/'
    for data, name in zip(data_list, names):
        embedding_list = []
        for sent in data:
            embedding_list.append(sent_embedding_generator(sent).cpu().numpy().tolist())
        with open(path + name +'_embeddings', 'w') as f:
            json.dump(embedding_list, f)
