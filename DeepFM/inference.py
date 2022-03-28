import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(saved_model, input_dims, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        input_dims = input_dims,
        embedding_dim = 10, 
        mlp_dims = [30, 20, 10],
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data = pd.read_csv(data_dir + '/eval/UC with Side-Information.csv')
    dataset = TestDataset(data_dir, data)

    input_dims = dataset.get_input_dimensions()
    model = load_model(model_dir, input_dims, device).to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    
    preds = []
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)

            outs = model(inputs)
            preds.extend(outs.cpu().numpy())

    print(f'Inference Done!')

    print(f'Creating Submission File..')
    data['preds'] = preds

    data = dataset.inverse_mapping(data)

    user_group_dfs = list(data.groupby('user'))
    submission = []
    for _, user_df in user_group_dfs:
        top10_preds = user_df.nlargest(10, ['preds'])
        submission.append(top10_preds[['user', 'item']])
    submission = pd.concat(submission, axis = 0, sort=False)
    
    submission.to_csv(os.path.join(output_dir, f'output.csv'), index=False)

    print('*** Finished ***')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='DeepFM', help='model type (default: DeepFM)')

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