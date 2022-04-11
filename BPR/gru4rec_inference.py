import argparse
import os
from importlib import import_module
import numpy as np

import pandas as pd
import torch
from torch.utils.data import DataLoader
import json

from dataset import GRU4RECDataset
from dataset import GRU4RECDataLoader


def zero_based_mapping(data) :
    with open('/opt/ml/movie-recommendation/data/train/zero_mapping.json', 'r') as f:
        dict_data= json.load(f)

    n_user = len(dict_data['user'])
    n_item = len(dict_data['item'])

    data['user']  = data['user'].map(lambda x : dict_data['user'][str(x)])
    data['item']  = data['item'].map(lambda x : dict_data['item'][str(x)])
    
    return data, n_user, n_item


def inverse_mapping(data):   
    with open('/opt/ml/movie-recommendation/data/train/zero_mapping.json', 'r') as f:
        dict_data= json.load(f)

    inv_user_map = {v:int(k) for k,v in dict_data['user'].items()}
    data['user'] = data['user'].map(lambda x : inv_user_map[x])

    inv_item_map = {v:int(k) for k,v in dict_data['item'].items()}
    data['item'] = data['item'].map(lambda x : inv_item_map[x])

    return data


def reset_hidden(hidden, mask):
    """Helper function that resets hidden state when some sessions terminate"""
    if len(mask) != 0:
        hidden[:, mask, :] = 0
    return hidden


def load_model(saved_model, n_item, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls( 
        input_size = n_item,
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print('Loading Dataset ...', end=' ')
    dataset = GRU4RECDataset(data_dir+'eval/GRU4REC.csv')

    model = load_model(model_dir, len(dataset.items), device)
    model.eval()

    loader = GRU4RECDataLoader(dataset, args.batch_size)
    print('Done!!')

    print("Calculating inference results...", end=' ')
    with open(data_dir+'train/session2user.json', 'r') as f:
        session2user = json.load(f)
    s2u = {int(k):v for k,v in session2user['s2u'].items()}
    
    outputs = torch.zeros([31360, 6807])
    outputs = outputs.to(device)

    with torch.no_grad():
        model.eval()
        
        hidden = model.init_hidden()
        for input, _, mask in loader:
            input = input.to(device)

            hidden = reset_hidden(hidden, mask).detach()
            logit, hidden = model(input, hidden)

            if len(mask) != 0 :
                outputs[list(map(s2u.get, mask)),:] += logit[mask]
    
    outputs = outputs.detach().cpu().numpy()
    preds = np.zeros((outputs.shape[0], 10))

    train_df = pd.read_csv(os.path.join(data_dir+'train', 'train_ratings.csv'))
    train_df,_,_ = zero_based_mapping(train_df)
    total_items = set(train_df['item'])
    observed_items_per_user = list(train_df.groupby('user')['item'])
    unobserved_dict = dict()
    for user, observed_items in observed_items_per_user:
        unobserved_dict[user] = list(total_items - set(observed_items))

    for i in range(outputs.shape[0]):
        indices = np.argpartition(outputs[i,unobserved_dict[i]], -10)[-10:]
        preds[i] = indices

    print('Done!!')

    print('Creating Submission File...', end=' ')
    submission = dict()

    submission['user'] = np.repeat(np.arange(preds.shape[0]), 10)
    submission['item'] = preds.reshape(-1)

    submission = pd.DataFrame(submission)
    submission = inverse_mapping(submission)
    submission.sort_values(by='user', inplace=True)
    submission.to_csv(os.path.join(output_dir, 'output.csv'), index=False)
    print('Done!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='GRU4REC', help='model type (default: DeepFM)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/movie-recommendation/data/'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)