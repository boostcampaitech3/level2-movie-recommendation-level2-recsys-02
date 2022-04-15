import os
import dgl
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm, tqdm_pandas, trange


def remap_id(id_lst, char) :
    id_lst.sort()
    id_to_idx, idx_to_id = dict(), dict()
    for index, value in enumerate(id_lst) :
        value = char + str(value)
        id_to_idx[value] = index
        idx_to_id[index] = value
    return id_to_idx, idx_to_id 

def preprocess(args) :
    rating_df = pd.read_csv(os.path.join(args.path, 'train_ratings.csv'))
    rating_df.head()
    
    user_id2idx, user_idx2id = remap_id(rating_df['user'].unique(), 'u')
    item_id2idx, item_idx2id = remap_id(rating_df['item'].unique(), 'i')
    
    rating_df['user'] = rating_df['user'].apply(lambda x: user_id2idx[f"u{x}"])
    rating_df['item'] = rating_df['item'].apply(lambda x: item_id2idx[f"i{x}"])
    
    graph = consrtruct_graph(rating_df)
    create_metapath(args, graph, item_idx2id)
    return rating_df, user_id2idx, user_idx2id, item_id2idx, item_idx2id

def consrtruct_graph(df) :
    hg = dgl.heterograph({
            ('user', 'ui', 'item') : (list(df['user']), list(df['item'])),
            ('item', 'iu', 'user') : (list(df['item']), list(df['user']))})
    return hg

def create_metapath(args, graph, item_idx2id) :
    output_file = open(os.path.join(args.path, 'metapath.txt'), "w")
    for user_idx in trange(graph.number_of_nodes('user')):
        traces, _ = dgl.sampling.random_walk(
            graph, [user_idx] * args.num_walks_per_node, metapath=['ui', 'iu'] * args.walk_length)

        for tr in traces:
            tr = tr[tr[:,]!=-1]

            outline = ''
            for i in range(len(tr)) :
                if i % 2 == 1 :
                    outline += item_idx2id[int(tr[i])] + ' '
            print(outline, file= output_file)
