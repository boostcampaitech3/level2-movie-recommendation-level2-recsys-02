import numpy as np
import torch
from torch.utils.data import Dataset

class SequentialDataset(Dataset):
    def __init__(self, data, num_negative=10, is_training=False) :
        self.data = data[['user', 'item']]
        self.n_user = self.data['user'].nunique() 
        self.n_item = self.data['item'].nunique()
        self.num_negative = num_negative
        self.is_training = is_training
        
        self.user2seq = dict()
        user_item_sequence = list(self.data.groupby(by='user')['item'])
        for user, item_seq in user_item_sequence :
            self.user2seq[user] = list(item_seq)
        
        if not self.is_training :
            self.users = list(self.user2seq.keys())
            self.item_seqs = list(self.user2seq.values())

    def negative_sampling(self):
        assert self.is_training, 'no need to sampling when testing'
        negative_samples = []
        
        for u, i in self.data.values:
            for _ in range(self.num_negative):
                j = np.random.randint(self.n_item)
                while j in self.user2seq[u]:
                    j = np.random.randint(self.n_item)
                negative_samples.append([u, i, j])
        self.features = negative_samples

    def __len__(self):
        return self.num_negative * len(self.data) if self.is_training else self.n_user
    
    def __getitem__(self, idx):
        return {"user":torch.tensor(self.features[idx][0]), 
                "pos_item": torch.tensor(self.features[idx][1]),
                "neg_item": torch.tensor(self.features[idx][2]),
                "item_seq": torch.tensor(
                    list(set(self.user2seq[self.features[idx][0]]) - \
                    set([self.features[idx][1]]))
                ),
                "seq_len": torch.tensor(len(self.user2seq[self.features[idx][0]])-1),}\
                if self.is_training else \
                {"user":torch.tensor(self.users[idx]),
                "pos_item": torch.arange(0,self.n_item),
                "neg_item": torch.tensor([0]),
                "item_seq": torch.tensor(self.item_seqs[idx]),
                "seq_len": torch.tensor(len(self.item_seqs[idx])),}


class SequentialDatasetv2(Dataset):
    def __init__(self, data, num_negative=10, window_size=100, is_training=False) :
        self.data = data[['user', 'item']]
        self.n_user = self.data['user'].nunique() 
        self.n_item = self.data['item'].nunique()
        self.num_negative = num_negative
        self.is_training = is_training
        self.window_size = window_size//2
        
        self.user2seq = dict()
        user_item_sequence = list(self.data.groupby(by='user')['item'])
        for user, item_seq in user_item_sequence :
            self.user2seq[user] = list(item_seq)
        
        if not self.is_training :
            self.user_id = list(self.user2seq.keys())[0]
            self.item_seq = list(self.user2seq.values())[0]

    def negative_sampling(self):
        assert self.is_training, 'no need to sampling when testing'
        negative_samples = []
        
        for u, i in self.data.values:
            for _ in range(self.num_negative):
                j = np.random.randint(self.n_item)
                while j in self.user2seq[u]:
                    j = np.random.randint(self.n_item)
                negative_samples.append([u, i, j])
        self.features = negative_samples

    def __len__(self):
        return self.num_negative * len(self.data) if self.is_training else len(self.item_seq)
    
    def __getitem__(self, idx):
        if self.is_training :
            user = self.features[idx][0]
            pos_item = self.features[idx][1]
            neg_item = self.features[idx][2]
            user_seq = self.user2seq[user]
            item_seq = list()

            index = self.find_index(user_seq, pos_item)
            
            item_seq.extend(user_seq[index-self.window_size:index] if index >= self.window_size else user_seq[0:index])
            item_seq.extend(user_seq[index+1 : index+1+self.window_size])

            return {"user":torch.tensor(user),
                    "pos_item": torch.tensor(pos_item),
                    "neg_item": torch.tensor(neg_item),
                    "item_seq": torch.tensor(item_seq),
                    "seq_len": torch.tensor(len(item_seq)),}

        item_seq = self.item_seq[idx-self.window_size:idx+1+self.window_size] if idx >= self.window_size \
                else self.item_seq[0:idx+1+self.window_size]

        return {"user":torch.tensor(self.user_id),
                "pos_item": torch.arange(0,6807),
                "neg_item": torch.tensor([0]),
                "item_seq": torch.tensor(item_seq),
                "seq_len": torch.tensor(len(item_seq)),}
    
    def find_index(self, seq, item):
        index = 0 
        for i in seq:
            if i == item:
                break
            index += 1
        return index


class SequentialDatasetv3(Dataset):
    def __init__(self, data, num_negative=10, is_training=False) :
        self.data = data[['user', 'item']]
        self.n_user = self.data['user'].nunique() 
        self.n_item = self.data['item'].nunique()
        self.num_negative = num_negative
        self.is_training = is_training
        
        self.user2seq = dict()
        user_item_sequence = list(self.data.groupby(by='user')['item'])
        for user, item_seq in user_item_sequence :
            self.user2seq[user] = list(item_seq)
        
        if not self.is_training :
            self.user = list(self.user2seq.keys())[0]
            self.item_seq = list(self.user2seq.values())[0]

    def negative_sampling(self):
        assert self.is_training, 'no need to sampling when testing'
        negative_samples = []
        
        for u, i in self.data.values:
            for _ in range(self.num_negative):
                j = np.random.randint(self.n_item)
                while j in self.user2seq[u]:
                    j = np.random.randint(self.n_item)
                negative_samples.append([u, i, j])
        self.features = negative_samples

    def __len__(self):
        return self.num_negative * len(self.data) if self.is_training else len(self.item_seq)
    
    def __getitem__(self, idx):
        if self.is_training :
            user = self.features[idx][0]
            pos_item = self.features[idx][1]
            neg_item = self.features[idx][2]
            user_seq = self.user2seq[user]
            item_seq = list()

            for item in user_seq :
                if item == pos_item :
                    break
                item_seq.append(item)
            
            if len(item_seq) == 0 :
                item_seq.append(user_seq[1])
        
        return {"user":torch.tensor(user),
                "pos_item": torch.tensor(pos_item),
                "neg_item": torch.tensor(neg_item),
                "item_seq": torch.tensor(item_seq),
                "seq_len": torch.tensor(len(item_seq)),}\
                if self.is_training else \
                {"user":torch.tensor(self.user),
                "pos_item": torch.arange(0,self.n_item),
                "neg_item": torch.tensor([0]),
                "item_seq": torch.tensor(self.item_seq[:idx+1] if idx>0 else self.item_seq[:2]),
                "seq_len": torch.tensor(idx+1 if idx>0 else 2),}