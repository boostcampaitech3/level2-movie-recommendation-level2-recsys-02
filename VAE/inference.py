import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import VAEDataLoader
from tqdm import tqdm

from model import MultiVAE


def load_model(saved_model, n_item, device):
    model = MultiVAE( 
        p_dims = [600, 1200, n_item],
    )
    model.load_state_dict(torch.load(saved_model, map_location=device))
    return model


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print('Loading Dataset ...', end=' ')

    unique_uid = list()
    with open(os.path.join(data_dir+'pro_sg', 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())

    unique_sid = list()
    with open(os.path.join(data_dir+'pro_sg', 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    loader = VAEDataLoader(data_dir)
    test_data = loader.load_data('test')

    input_x = naive_sparse2tensor(test_data).to(device)

    print('Done!!')

    n_splits = 20
    output_sum = torch.zeros(31360,6807).to(device)
    dropout_list = [0.3, 0.4, 0.5, 0.6, 0.7]

    print("Calculating inference results...", end=' ')

    for dropout in dropout_list:
        for k in range(n_splits) :
            model = load_model(f'{model_dir}/{dropout}/cv{k+1}.pth', 6807, device).to(device)
            model.eval()

            with torch.no_grad():
                output_sum += model(input_x)[0] / (n_splits*len(dropout_list))
    
    torch.save(output_sum, './pby_result.pt')
    
    prediction = output_sum - input_x * 99999
    ranking = torch.topk(prediction, k=10)[1]

    show2id = dict((i,sid) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((i,pid) for (i, pid) in enumerate(unique_uid))

    sub_u, sub_i = [], []
    for target_u in range(0, 31360) :
        target_i = ranking[target_u]
        for target in target_i:
            sub_u.append(int(profile2id[target_u]))
            sub_i.append(show2id[int(target)])

    print('Done!!')

    print('Creating Submission File...', end=' ')

    submission = {"user" : sub_u, "item" : sub_i}
    submission_df = pd.DataFrame(submission)
    submission_df.sort_values('user', inplace=True)
    submission_df.to_csv(f'{output_dir}/final-2.csv', index=False)

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