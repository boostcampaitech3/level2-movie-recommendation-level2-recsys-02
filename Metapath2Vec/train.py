import os
import argparse
from preprocessing import *
from metapath2vec import * 
from inference import *

def train(args) :
    # preprocessing
    rating_df, user_id2idx, user_idx2id, item_id2idx, item_idx2id = preprocess(args)
    
    # train
    m2v = Metapath2VecTrainer(os.path.join(args.path, args.metapath_file))
    m2v.train()
    m2v.skip_gram_model.save_embedding(m2v.data.id2word, args.save_path)
    
    # inference
    inference(args, m2v, user_idx2id)
    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--model', type=str, default='Metapath2Vec', help='model type (default: Metapath2Vec)')
    parser.add_argument('--path', type=str, default='../data/train')
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--num_walks_per_node', type=int, default=1000)
    parser.add_argument('--metapath_file', type=str, default='metapath.txt')
    parser.add_argument('--save_path', type=str, default='metapath2vec.pkl')
    parser.add_argument('--submission_file', type=str, default='m2v_submission.csv')
    
    args = parser.parse_args()
    print(args)
    
    train(args)
    