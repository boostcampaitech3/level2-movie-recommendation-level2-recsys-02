from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split


class RatingDataset(Dataset):
    def __init__(self, data_dir):
        data = pd.read_csv(data_dir + '/train/context-aware/ratings_with_genres.csv')

        self.n_user = data.loc[:,'user'].nunique()
        self.n_item = data.loc[:,'item'].nunique()
        self.n_genre = data.loc[:,'genre'].nunique()

        user_col = torch.tensor(np.array(data.loc[:,'user']))
        item_col = torch.tensor(np.array(data.loc[:,'item']))
        genre_col = torch.tensor(np.array(data.loc[:,'genre']))

        offsets = [0, self.n_user, self.n_user+self.n_item]
        for col, offset in zip([user_col, item_col, genre_col], offsets):
            col += offset

        self.X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1)], dim=1).long()
        self.y = torch.tensor(list(data.loc[:,'rating'])).long()
        self.data_len = len(data)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.data_len

    def split_dataset(self, train_ratio=0.9) -> Tuple[Subset, Subset]:
        train_size = int(train_ratio * self.data_len)
        test_size = self.data_len - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        
        return train_dataset, test_dataset

    def get_input_dimensions(self):
        return [self.n_user, self.n_item, self.n_genre]


class TestDataset(Dataset):
    def __init__(self, data_dir, data):
        train_df = pd.read_csv(data_dir + '/train/context-aware/ratings_with_genres.csv')
        
        self.n_user = train_df.loc[:,'user'].nunique()
        self.n_item = train_df.loc[:,'item'].nunique()
        self.n_genre = train_df.loc[:,'genre'].nunique()

        del train_df

        user_col = torch.tensor(np.array(data.loc[:,'user']))
        item_col = torch.tensor(np.array(data.loc[:,'item']))
        genre_col = torch.tensor(np.array(data.loc[:,'genre']))

        offsets = [0, self.n_user, self.n_user+self.n_item]
        for col, offset in zip([user_col, item_col, genre_col], offsets):
            col += offset

        self.X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1)], dim=1).long()
        self.data_len = len(data)
    
    def __getitem__(self, index):
        return self.X[index]
    
    def __len__(self):
        return self.data_len

    def get_input_dimensions(self):
        return [self.n_user, self.n_item, self.n_genre]