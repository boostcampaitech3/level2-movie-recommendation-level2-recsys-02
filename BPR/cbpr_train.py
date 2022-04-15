import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
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


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def get_valid_score(all_preds, data) :
    all_preds = torch.cat(all_preds).detach().cpu().numpy()
    data['preds'] = all_preds

    user_group_dfs = list(data.groupby(by='user'))

    predicted = []
    actual = []

    for _, user_df in user_group_dfs :
        recommends = np.array(user_df.nlargest(10, ['preds'])['item'])
        predicted.append(recommends)
        ground_truth = np.array(user_df[user_df['rating'] == 1]['item'])
        actual.append(ground_truth)
    
    return recall_at_k(actual, predicted, 10)


def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    print('Loading Dataset ...', end=' ')
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    
    train_set = dataset_module(
        data_path=data_dir+'train.csv',
        num_negative=args.num_negative
    )
    
    val_set = dataset_module(
        data_path=data_dir+'valid.csv', 
        is_training=False
    )

    # -- data_loader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
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

    best_val_recall_k = 0
    counter = 0
    patience = args.patience
    accumulation_steps = args.accumulation_steps
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0

        print('Negative Sampling ...', end=' ')
        train_loader.dataset.negative_sampling()
        print('Done!!')

        for idx, train_batch in enumerate(train_loader):
            train_batch = tuple(t.to(device) for t in train_batch)
            user, item_i, item_j, context_i, context_j = train_batch

            prediction_i, prediction_j = model(user, item_i, item_j, context_i, context_j)
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

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()

            all_preds = []
            for val_batch in val_loader:
                val_batch = tuple(t.to(device) for t in val_batch)
                user, item_i, item_j, context_i, context_j = val_batch

                prediction_i, prediction_j = model(user, item_i, item_j, context_i, context_j)

                all_preds.append(prediction_i)
            
            val_recall_k = get_valid_score(all_preds, val_set.data)

            if val_recall_k - 0.0001 > best_val_recall_k:
                counter = 0
            else :
                counter += 1

            if val_recall_k > best_val_recall_k:
                print("New best model for val accuracy! saving the model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_recall_k = val_recall_k
            
            # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
            if counter > patience:
                print("Early Stopping...")
                break
            print(
                f"[Val] recall@10 : {val_recall_k:4.4%} || "
                f"best recall@10 : {best_val_recall_k:4.4%} || "
                f"counter/patience : {counter}/{patience} "
            )
            logger.add_scalar("Val/accuracy", val_recall_k, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='ContextualBPRDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1024, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='ContextualBPRv3', help='model type (default: BPR)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=2000, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--factor_num', type=int, default=10, help='number of factors (default: 10)')
    parser.add_argument('--num_negative', type=int, default=10, help='number of negative samples (default: 10)')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping (default: 5)')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='batch accumulation step (default: 1)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/movie-recommendation/data/train/bpr/'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)