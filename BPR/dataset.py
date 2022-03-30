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


class BPRDataset(Dataset):
    def __init__(self, data_path, num_negative=5, is_training=True, all_cases=False):
        super(BPRDataset, self).__init__()

        if all_cases :
            self.data = self.get_all_cases()
        else :
            self.data = pd.read_csv(data_path)
            if 'rating' not in self.data.columns :
                self.data = self.data[['user', 'item']].sort_values(by=['user'])
            else :
                self.data = self.data[['user', 'item', 'rating']].sort_values(by=['user'])

        self.data, self.n_user, self.n_item= zero_based_mapping(self.data)
        
        if is_training :
            self.get_sparse_matrix()

        self.num_negative = num_negative
        self.is_training = is_training
        self.features = self.data.values

    def negative_sampling(self):
        assert self.is_training, 'no need to sampling when testing'
        negative_samples = []
        
        for u, i in self.data.values:
            for _ in range(self.num_negative):
                j = np.random.randint(self.n_item)
                while (u, j) in self.train_matrix:
                    j = np.random.randint(self.n_item)
                negative_samples.append([u, i, j])
        
        self.features = negative_samples
    
    def __len__(self):
        if self.is_training :
            return self.num_negative * len(self.data)
        return len(self.data)

    def __getitem__(self, idx):
        user = self.features[idx][0]
        item_i = self.features[idx][1]
        item_j = self.features[idx][2] if \
				self.is_training else self.features[idx][1]
        return user, item_i, item_j 
    
    def get_sparse_matrix(self):
        train_matrix = dok_matrix((self.n_user, self.n_item), dtype=np.float32)
        for u, i in self.data.values:
            train_matrix[u, i] = 1.0
        self.train_matrix = train_matrix
    
    def get_all_cases(self):
        # Extract Top Most Popular movies
        train_df = pd.read_csv('/opt/ml/movie-recommendation/data/train/train_ratings.csv')

        items = set(train_df['item'])
        observed_items_per_user = list(train_df.groupby('user')['item'])

        # 각 유저마다 안본 영화만 선택
        unseen_items_dfs = list()

        for user, observed_items in observed_items_per_user:
            observed_items = set(observed_items)
            unseen_item = list(items - observed_items)

            user_id = [user]*len(unseen_item)
            unseen_items_dfs.append(pd.DataFrame(zip(user_id,unseen_item), columns=['user','item']))

        test_df = pd.concat(unseen_items_dfs, axis = 0, sort=False)

        test_df = test_df.sort_values(by=['user'])
        test_df.reset_index(drop=True, inplace=True)
        return test_df