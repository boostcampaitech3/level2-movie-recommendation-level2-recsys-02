import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dataset import make_batch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


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


def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    print('Loading Dataset ...', end=' ')
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    
    df = pd.read_csv(data_dir+'train_ratings.csv')
    df, user_to_id, id_to_user, movie_to_id, id_to_movie = preprocess(df)

    train_set = dataset_module(df, num_negative=10, is_training=True)
    train_loader = DataLoader(
        train_set,
        batch_size=1024,
        shuffle=True, 
        drop_last=True,
        collate_fn=make_batch,
        pin_memory=True,
        num_workers=4
    )

    print('Done!!')

    # -- model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        user_num = train_set.n_user, 
        item_num = train_set.n_item,
        factor_num = args.factor_num, 
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    opt_module = getattr(import_module("torch.optim"), args.optimizer) 
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    accumulation_steps = args.accumulation_steps
    
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0

        print('Negative Sampling ...', end=' ')
        train_loader.dataset.negative_sampling()
        print('Done!!')

        for idx, train_batch in enumerate(train_loader):
            user, pos_item, neg_item, item_seq, seq_len =\
                (v.to(device) for _,v in train_batch.items())

            prediction_i = model(user, pos_item, item_seq, seq_len)
            prediction_j = model(user, neg_item, item_seq, seq_len)
            loss =- (prediction_i - prediction_j).sigmoid().log().sum()

            loss.backward()
            if (idx+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_value += loss.item()

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.7} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)

                loss_value = 0
        
        torch.save(model.module.state_dict(), f"{save_dir}/epoch-{epoch}.pth")
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--dataset', type=str, default='SequentialDatasetv3', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 64)')
    # parser.add_argument('--valid_batch_size', type=int, default=1024, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='FPMC', help='model type (default: BPR)')
    parser.add_argument('--factor_num', type=int, default=32, help='number of factors (default: 10)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--num_negative', type=int, default=10, help='number of negative samples (default: 10)')
    # parser.add_argument('--patience', type=int, default=5, help='patience for early stopping (default: 5)')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='batch accumulation step (default: 1)')
    parser.add_argument('--log_interval', type=int, default=1000, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/movie-recommendation/data/train/'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)