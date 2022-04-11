import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import make_batch


def preprocess(df) :
    df = df.sort_values(['user', 'time'], ascending = [True, True])

    users = df['user'].unique()
    user_to_id = dict(zip(users, range(len(users))))
    id_to_user = {v: k for k, v in user_to_id.items()}
    
    movies = df['item'].unique()
    movie_to_id = dict(zip(movies, range(len(movies))))
    id_to_movie = {v: k for k, v in movie_to_id.items()}
    
    df['user'] = df['user'].apply(lambda x : user_to_id[x])
    df['item'] = df['item'].apply(lambda x : movie_to_id[x])

    return df, user_to_id, id_to_user, movie_to_id, id_to_movie
    

def load_model(saved_model, n_user, n_item, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        user_num = n_user, 
        item_num = n_item,
        factor_num = 32, 
    )

    model_path = os.path.join(saved_model, 'epoch-9.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, 31360, 6807, device).to(device)

    print('Loading Dataset ...', end=' ')

    df = pd.read_csv(data_dir+'train_ratings.csv')
    df, user_to_id, id_to_user, movie_to_id, id_to_movie = preprocess(df)

    user_item_dfs = df.groupby('user')
    dataset_module = getattr(import_module("dataset"), args.dataset)

    print('Done!!')


    print("Calculating inference results...", end=' ')

    sub_u = []
    sub_i = []

    for user_id, item_df in user_item_dfs:
        test_dataset = dataset_module(item_df)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False, 
            drop_last=False,
            collate_fn=make_batch,
        )
        with torch.no_grad():
            model.eval()
            
            prediction = torch.zeros(6807).to(device)

            for batch in test_loader:
                user, pos_item, neg_item, item_seq, seq_len =\
                    (v.to(device) for _,v in batch.items())

                output = model.predict(user, pos_item, item_seq, seq_len)
                prediction += output.sum(axis=0)

            ranking = torch.topk(prediction, len(prediction))[1]
        
        pred = []
        for item_id in ranking :
            if item_id in test_dataset.item_seq :
                continue
            u = id_to_user[int(user_id)]
            i = id_to_movie[int(item_id)]
            sub_u.append(u)
            sub_i.append(i)
            pred.append(i)
            if len(pred) == 10 :
                break

    print('Done!!')

    print('Creating Submission File...', end=' ')

    submission = {"user" : sub_u, "item" : sub_i}
    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(f'{output_dir}/output.csv', index=False)

    print('Done!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for validing (default: 1000)')
    parser.add_argument('--dataset', type=str, default='SequentialDatasetv2', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--model', type=str, default='FPMC', help='model type (default: DeepFM)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/movie-recommendation/data/train/'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)