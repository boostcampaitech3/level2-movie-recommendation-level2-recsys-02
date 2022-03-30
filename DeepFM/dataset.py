from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split
from scipy.sparse import dok_matrix


class DeepFMDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

        self.n_user = self.data.loc[:,'user'].nunique()
        self.n_item = self.data.loc[:,'item'].nunique()
        self.n_year = self.data.loc[:,'year'].nunique()
        self.n_genre = 18

        self.zero_based_mapping()

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
    
    def zero_based_mapping(self):
        # user, item을 zero-based index로 mapping
        users = list(set(self.data.loc[:,'user']))
        items =  list(set((self.data.loc[:, 'item'])))

        if len(users)-1 != max(users):
            users_dict = {users[i]: i for i in range(len(users))}
            self.data['user']  = self.data['user'].map(lambda x : users_dict[x])
            
        if len(items)-1 != max(items):
            items_dict = {items[i]: i for i in range(len(items))}
            self.data['item']  = self.data['item'].map(lambda x : items_dict[x])


class DeepFMTestDataset(Dataset):
    def __init__(self, data_dir, data):
        train_df = pd.read_csv(data_dir+'/train/context-aware/Ratings with Side-Information.csv')

        self.n_user = train_df.loc[:,'user'].nunique()
        self.n_item = train_df.loc[:,'item'].nunique()
        self.n_year = train_df.loc[:,'year'].nunique()
        self.n_genre = 18
        self.data = data

        self.zero_based_mapping(train_df)

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
    
    def zero_based_mapping(self, train_df):
        # user, item을 zero-based index로 mapping
        users = list(set(train_df.loc[:,'user']))
        items =  list(set(train_df.loc[:, 'item']))

        if len(users)-1 != max(users):
            self.users_dict = {users[i]: i for i in range(len(users))}
            self.data['user']  = self.data['user'].map(lambda x : self.users_dict[x])

        if len(items)-1 != max(items):
            self.items_dict = {items[i]: i for i in range(len(items))}
            self.data['item']  = self.data['item'].map(lambda x : self.items_dict[x])
    
    def inverse_mapping(self, data_df):   
        inv_user_map = {v: k for k, v in self.users_dict.items()}
        data_df['user']  = data_df['user'].map(lambda x : inv_user_map[x])

        inv_item_map = {v: k for k, v in self.items_dict.items()}
        data_df['item'] = data_df['item'].map(lambda x : inv_item_map[x])

        return data_df


class BPRDataset(Dataset):
	def __init__(self, data_path, num_negative=5, is_training=True):
		super(BPRDataset, self).__init__()
		
		self.data = pd.read_csv(data_path)

		if is_training :
			self.data = self.data[['user', 'item']].sort_values(by=['user'])
		else :
			self.data = self.data[['user', 'item', 'rating']].sort_values(by=['user'])

		self.zero_based_mapping()
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
		return self.num_negative * len(self.data) if \
				self.is_training else len(self.data)

	def __getitem__(self, idx):
		user = self.features[idx][0]
		item_i = self.features[idx][1]
		item_j = self.features[idx][2] if \
				self.is_training else self.features[idx][1]
		return user, item_i, item_j 
	
	def zero_based_mapping(self) :
		users = list(set(self.data.loc[:,'user']))
		items =  list(set(self.data.loc[:, 'item']))

		self.n_user = len(users)
		self.n_item = len(items)

		# user, item을 zero-based index로 mapping
		if self.n_user-1 != max(users):
			users_dict = {users[i]: i for i in range(len(users))}
			self.data['user']  = self.data['user'].map(lambda x : users_dict[x])

		if self.n_item-1 != max(items):
			items_dict = {items[i]: i for i in range(len(items))}
			self.data['item']  = self.data['item'].map(lambda x : items_dict[x])
	
	def get_sparse_matrix(self):
		train_matrix = dok_matrix((self.n_user, self.n_item), dtype=np.float32)
		for u, i in self.data.values:
			train_matrix[u, i] = 1.0
		self.train_matrix = train_matrix