import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BPRDataset

def inverse_mapping(data):   
    train_df = pd.read_csv('/opt/ml/movie-recommendation/data/train/train_ratings.csv')

    users = list(set(train_df.loc[:,'user']))
    items =  list(set(train_df.loc[:, 'item']))

    inv_user_map = {i: users[i] for i in range(len(users))}
    data['user'] = data['user'].map(lambda x : inv_user_map[x])

    inv_item_map = {i: items[i] for i in range(len(items))}
    data['item'] = data['item'].map(lambda x : inv_item_map[x])

    return data
    

def load_model(saved_model, n_user, n_item, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        user_num = n_user, 
        item_num = n_item,
        factor_num = 10, 
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print('Loading Dataset ...', end=' ')
    dataset = BPRDataset(data_dir + '/eval/Unobserved Cases.csv', is_training=False)

    model = load_model(model_dir, 31360, 6807, device).to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    print('Done!!')

    print("Calculating inference results...", end=' ')
    
    with torch.no_grad():
        model.eval()

        preds = []
        
        for batch in loader:
            user, item_i, item_j = batch
            user = user.to(device)
            item_i = item_i.to(device)
            item_j = item_j.to(device)

            prediction_i, _ = model(user, item_i, item_j)

            preds.append(prediction_i)
        
        preds = torch.cat(preds).detach().cpu().numpy()

    print('Done!!')

    print('Creating Submission File...', end=' ')
    dataset.data['preds'] = preds

    dataset.data = inverse_mapping(dataset.data)

    user_group_dfs = list(dataset.data.groupby('user'))
    submission = []
    for _, user_df in user_group_dfs:
        top10_preds = user_df.nlargest(10, ['preds'])
        submission.append(top10_preds[['user', 'item']])
    submission = pd.concat(submission, axis = 0, sort=False)
    
    submission.to_csv(os.path.join(output_dir, 'output.csv'), index=False)

    print('Done!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BPR', help='model type (default: DeepFM)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/movie-recommendation/data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)