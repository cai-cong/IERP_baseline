import pandas
import math
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import json
import glob
from torch.utils.data import DataLoader

class EmotionDataset(Dataset):
    def __init__(self, partition, data):
        features, labels = data[partition]['feature'], data[partition]['label']
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

    def __len__(self):
        return len(self.features)

def collate_fn(data):

    for i in range(0,len(data)):
        data[i] = (i,data[i])
    data.sort(key=lambda x:x[1][0].shape[0],reverse=True)

    ids, tmp_features_labels = zip(*data)
    features_tmp,labels = zip(*tmp_features_labels)
    labels = torch.FloatTensor(np.array(labels))

    feature_dim = len(features_tmp[0][0])
    lengths = [len(f) for f in features_tmp]
    max_length = max(lengths)
    features = torch.zeros((len(features_tmp), max_length,  feature_dim)).float()

    for i ,feature in enumerate(features_tmp):
        end = lengths[i]
        feature = torch.FloatTensor(np.array(feature))
        features[i,:end, :]= feature[:end, :]
    return features,labels, lengths

def get_dataloader(args,data):
    batch_size = args.batch_size

    train_dataset = EmotionDataset("train", data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

    dev_dataset  = EmotionDataset("val", data)
    dev_dataloader = DataLoader(dataset=dev_dataset,batch_size=batch_size, shuffle=False,collate_fn=collate_fn)

    test_dataset  = EmotionDataset("test", data)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False,collate_fn=collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader