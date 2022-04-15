from typing import Tuple

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split
from scipy.sparse import dok_matrix

def zero_based_mapping(data) :
    train_df = pd.read_csv('/opt/ml/movie-recommendation/data/train/train_ratings.csv')

    users = list(set(train_df.loc[:,'user']))
    items =  list(set(train_df.loc[:, 'item']))

    n_user = len(users)
    n_item = len(items)

    # user, item을 zero-based index로 mapping
    if n_user-1 != max(users):
        users_dict = {users[i]: i for i in range(len(users))}
        data['user']  = data['user'].map(lambda x : users_dict[x])
        
    if n_item-1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        data['item']  = data['item'].map(lambda x : items_dict[x])
    
    return data, n_user, n_item


class FMDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

        self.data, _, _ = zero_based_mapping(self.data)
        self.attributes = self.get_item_attributes()

        self.X = torch.tensor(np.array(self.data.loc[:, ['user', 'item']])).long()
        self.y = torch.tensor(np.array(self.data.loc[:, 'rating'])).long()

    def __getitem__(self, index):
        item_i = self.X[index, 1]
        X = torch.cat([self.X[index], self.attributes[item_i]])
        return X, self.y[index]

    def __len__(self):
        return len(self.data)

    def split_dataset(self, train_ratio=0.9) -> Tuple[Subset, Subset]:
        train_size = int(train_ratio * len(self.data))
        test_size = len(self.data) - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        
        return train_dataset, test_dataset
    
    def get_item_attributes(self):
        data_dir = '/opt/ml/movie-recommendation/data/train/'

        with open(data_dir+'item2attributes.json', 'r') as f:
            item2attributes = json.load(f)

        attributes = []

        for item in range(6807):    
            attribute = [0] * 18
            now_attribute = item2attributes[str(item)]
            for a in now_attribute[1:]:
                attribute[a] = 1
            attributes.append([now_attribute[0]]+attribute)
        
        return torch.tensor(attributes)