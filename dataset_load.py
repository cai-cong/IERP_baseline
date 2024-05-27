import glob
import os
import time
import numpy as np
import pandas as pd
import pickle


def get_data_partition(partition_file):
    vid2partition, partition2vid = {}, {}
    df = pd.read_csv(partition_file)

    for row in df.values:

        vid, partition = str("%03d"%row[0]), row[1]
        vid2partition[vid] = partition
        if partition not in partition2vid:
            partition2vid[partition] = []
        if vid not in partition2vid[partition]:
            partition2vid[partition].append(vid)
    return vid2partition, partition2vid

def load_data(args):
    feature_path = os.path.join(args.dataset_file_path,"features",args.feature_set)
    label_path = os.path.join(args.dataset_file_path,"labels")

    data_file_name = f'data_{args.feature_set}_{args.fea_dim}.pkl'
    data_file = os.path.join("./data_cache", data_file_name)

    if os.path.exists(data_file):  # check if file of preprocessed data exists
        print(f'Find cached data "{os.path.basename(data_file)}".')
        data = pickle.load(open(data_file, 'rb'))
        return data

    print('Constructing data from scratch ...')
    data = {'train': {'feature': [], 'label': []},
            'val': {'feature': [], 'label': []},
            'test': {'feature': [], 'label': []}}
    vid2partition, partition2vid = get_data_partition(os.path.join(args.dataset_file_path,"partition.csv"))

    for partition, vids in partition2vid.items():
        for vid in vids:
            dir = os.path.join(feature_path, vid)
            for file in sorted(os.listdir(dir)):

                if file.endswith("csv"):
                    feature = pd.read_csv(os.path.join(dir, file), header=None).to_numpy()
                elif file.endswith("npy"):
                    feature = np.load(os.path.join(dir, file))
                number = int(file[4])
                label_file = os.path.join(label_path, vid + '.csv')
                label = (pd.read_csv(label_file).iloc[:,1:]).to_numpy()

                data[partition]['label'].append(label[number-1,:])
                data[partition]['feature'].append(feature)

    if not os.path.exists("./data_cache"):
        os.mkdir("./data_cache")
    pickle.dump(data, open(data_file, 'wb'))

    return data