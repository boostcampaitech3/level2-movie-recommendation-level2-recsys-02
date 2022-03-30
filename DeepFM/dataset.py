from typing import Tuple

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

class DeepFMDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

        self.n_year = self.data.loc[:,'year'].nunique()
        self.n_genre = 18

        self.data, self.n_user, self.n_item = zero_based_mapping(self.data)

        user_col = torch.tensor(np.array(self.data.loc[:,'user']))
        item_col = torch.tensor(np.array(self.data.loc[:,'item']))
        year_col = torch.tensor(np.array(self.data.loc[:,'year']))

        multi_hot_cols = self.data.columns.difference(['user', 'item', 'year','rating'])
        genre_col = torch.tensor(np.array(self.data.loc[:,multi_hot_cols]))

        offsets = [0, self.n_user, self.n_user+self.n_item]
        for col, offset in zip([user_col, item_col, year_col], offsets):
            col += offset

        self.X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), year_col.unsqueeze(1), genre_col], dim=1).long()
        self.y = torch.tensor(np.array(self.data.loc[:,'rating'])).long()

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.data)

    def split_dataset(self, train_ratio=0.9) -> Tuple[Subset, Subset]:
        train_size = int(train_ratio * len(self.data))
        test_size = len(self.data) - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        
        return train_dataset, test_dataset

    def get_input_dimensions(self):
        return [self.n_user, self.n_item, self.n_year, self.n_genre]


class DeepFMTestDataset(Dataset):
    def __init__(self, data_dir, data):
        train_df = pd.read_csv(data_dir+'/train/context-aware/Ratings with Side-Information.csv')

        self.n_year = train_df.loc[:,'year'].nunique()
        self.n_genre = 18
        
        self.data, self.n_user, self.n_item = zero_based_mapping(data)

        user_col = torch.tensor(np.array(self.data.loc[:,'user']))
        item_col = torch.tensor(np.array(self.data.loc[:,'item']))
        year_col = torch.tensor(np.array(self.data.loc[:,'year']))

        multi_hot_cols = self.data.columns.difference(['user', 'item', 'year','rating'])
        genre_col = torch.tensor(np.array(self.data.loc[:,multi_hot_cols]))

        offsets = [0, self.n_user, self.n_user+self.n_item]
        for col, offset in zip([user_col, item_col, year_col], offsets):
            col += offset

        self.X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), year_col.unsqueeze(1), genre_col], dim=1).long()
    
    def __getitem__(self, index):
        return self.X[index]
    
    def __len__(self):
        return len(self.data)

    def get_input_dimensions(self):
        return [self.n_user, self.n_item, self.n_year, self.n_genre]