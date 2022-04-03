import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, random_split, Subset
from scipy.sparse import dok_matrix
import bisect
import random
import torch
import pickle
from typing import Tuple


def zero_based_mapping(data) :
    with open('/opt/ml/movie-recommendation/data/train/zero_mapping.json', 'r') as f:
        dict_data= json.load(f)

    n_user = len(dict_data['user'])
    n_item = len(dict_data['item'])

    data['user']  = data['user'].map(lambda x : dict_data['user'][str(x)])
    data['item']  = data['item'].map(lambda x : dict_data['item'][str(x)])
    
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


class PopularBPRDataset(BPRDataset):
    def __init__(self, data_path, num_negative=10, is_training=True, all_cases=False):
        super(PopularBPRDataset, self).__init__(
            data_path=data_path,
            num_negative=num_negative,
            is_training=is_training,
            all_cases=all_cases
        )
        
        if is_training :
            self.popular_count()

    def popular_count(self):
        pop_count = np.squeeze(self.train_matrix.sum(axis=0).A)
        self.pop_count = np.log(pop_count + 1)
        self.pop_prob = self.pop_count / np.sum(self.pop_count)
        self.pop_cum_prob = self.pop_prob.cumsum()

    def negative_sampling(self):
        assert self.is_training, 'no need to sampling when testing'
        negative_samples = []
        
        for u, i in self.data.values:
            for _ in range(self.num_negative):
                j = bisect.bisect(self.pop_cum_prob, random.random())
                while (u, j) in self.train_matrix:
                    j = bisect.bisect(self.pop_cum_prob, random.random())
                negative_samples.append([u, i, j])

        self.features = negative_samples


class ContextualBPRDataset(BPRDataset):
    def __init__(self, data_path, num_negative=5, is_training=True, all_cases=False):
        super(ContextualBPRDataset, self).__init__(
            data_path=data_path,
            num_negative=num_negative,
            is_training=is_training,
            all_cases=all_cases
        )

        self.item_context = self.get_item_context()

    def __getitem__(self, idx):
        user = self.features[idx][0]
        item_i = self.features[idx][1]
        item_j = self.features[idx][2] if \
				self.is_training else self.features[idx][1]
        context_i = self.item_context[item_i]
        context_j = self.item_context[item_j]

        return user, item_i, item_j, context_i, context_j
    
    def get_item_context(self):
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
        
        return np.array(attributes)


class m2vBPRDataset(ContextualBPRDataset):
    def __init__(self, data_path, num_negative=5, is_training=True, all_cases=False):
        super(m2vBPRDataset, self).__init__(
            data_path=data_path,
            num_negative=num_negative,
            is_training=is_training,
            all_cases=all_cases
        )

        self.item_context = self.get_item_context()
    
    def get_item_context(self):
        m2v_dir = '/opt/ml/movie-recommendation/data/train/m2v/'

        with open(m2v_dir+'m2v_item2index.pkl', 'rb') as f :
            m2v_item2index = pickle.load(f)

        with open(m2v_dir+'m2v_item_emb.pkl', 'rb') as f :
            m2v_item_emb = pickle.load(f)

        attributes = np.zeros(shape=m2v_item_emb.shape)
        for np_index, item_id in m2v_item2index.items():
            attributes[item_id] = m2v_item_emb[np_index]
        
        return attributes


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