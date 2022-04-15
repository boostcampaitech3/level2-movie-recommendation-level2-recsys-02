import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import dgl
import torch
from tqdm import tqdm
import pickle

def get_nagative_items(data_array, all_items) :
    neg_items = dict()
    for user in tqdm(np.unique(data_array[:, 0])) :
        seen = set(data_array[data_array[:, 0]==user][:, 1])
        neg_item = all_items - seen
        neg_items[user] = list(neg_item)

    with open('./neg_items.pkl', 'wb') as f :
        pickle.dump(neg_items, f)
    return neg_items

def preprocess(data_path) :
    dtypes = {'user': np.int32, 'item': np.int32}
    data_train = pd.read_csv(data_path + 'train_ratings.csv')
    data_train.drop('time', inplace = True, axis = 1)
    
    users = data_train['user'].unique()
    movies = data_train['item'].unique()
    
    user_ids = range(len(users))
    movie_ids = range(len(users), len(users)+len(movies))
    
    num_user = users.shape[0]
    num_item = movies.shape[0]
    
    user_to_id = dict(zip(users, user_ids))
    movie_to_id = dict(zip(movies, movie_ids))
    id_to_user = {v: k for k, v in user_to_id.items()}
    id_to_movie = {v: k for k, v in movie_to_id.items()}
    
    data_train['user'] = data_train['user'].apply(lambda x : user_to_id[x])
    data_train['item'] = data_train['item'].apply(lambda x : movie_to_id[x])
    
    data_train, data_test = train_test_split(data_train, test_size=0.2)
    
    data_array_train = data_train.values.tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.values.tolist()
    data_array_test = np.array(data_array_test)
    
    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    train_graph = create_graph(data_array_train)
    return train_graph, id_to_movie, id_to_user, movie_to_id, id_to_movie, data_array, data_array_train, data_array_test, num_user, num_item, user_to_id
    
def create_graph(data_array_train) :
    u_nodes, v_nodes = [], []
    for user, movie in data_array_train :
        u_nodes.append(user)
        v_nodes.append(movie)

    
    train_graph = dgl.graph((torch.tensor(u_nodes), torch.tensor(v_nodes)))
    train_graph = dgl.to_bidirected(train_graph)
    
    return train_graph