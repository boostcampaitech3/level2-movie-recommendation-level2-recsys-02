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
from loss import create_criterion
from dataset import GRU4RECDataLoader


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


def reset_hidden(hidden, mask):
    """Helper function that resets hidden state when some sessions terminate"""
    if len(mask) != 0:
        hidden[:, mask, :] = 0
    return hidden


def get_recall(indices, targets, k=10): #recall --> wether next item in session is within top K=20 recommended items or not
    """
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """
    _, indices = torch.topk(indices, k, -1)
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall


def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    print('Loading Dataset ...', end=' ')
    dataset_module = getattr(import_module("dataset"), args.dataset)
    
    train_set = dataset_module(
        data_dir=data_dir,
    )
    
    valid_set = dataset_module(
        data_dir=data_dir,
        is_training=False
    )

    print('Done!!')

    # -- model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        input_size=len(train_set.items)
    ).to(device)

    # -- loss & metric
    opt_module = getattr(import_module("torch.optim"), args.optimizer) 
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    criterion = create_criterion(args.criterion)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_recall_k = 0
    best_val_loss = np.inf
    counter = 0
    patience = args.patience
    accumulation_steps = args.accumulation_steps
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        train_loader = GRU4RECDataLoader(train_set, batch_size=args.batch_size)
        
        hidden = model.init_hidden()
        for idx, (input, target, mask) in enumerate(train_loader):
            input = input.to(device)
            target = target.to(device)
            
            hidden = reset_hidden(hidden, mask).detach()
            logit, hidden = model(input, hidden)

            logit_sampled = logit[:, target.view(-1)]
            loss = criterion(logit_sampled)

            loss.backward()
            if (idx+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_value += loss.item()

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader.dataset.df) // train_loader.batch_size}) || "
                    f"training loss {train_loss:4.4} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader.dataset.df) // train_loader.batch_size + idx)

                loss_value = 0

        scheduler.step()

        # val loop
        print("Calculating validation results...")
        val_loss_items = []
        all_recalls = []
        valid_loader = GRU4RECDataLoader(valid_set, batch_size=args.valid_batch_size)

        model.eval()
        with torch.no_grad():
            hidden = model.init_hidden()
            for idx, (input, target, mask) in enumerate(valid_loader):
                input = input.to(device)
                target = target.to(device)
                logit, hidden = model(input, hidden)
                logit_sampled = logit[:, target.view(-1)]
                loss = criterion(logit_sampled)
                recall = get_recall(logit, target)

                val_loss_items.append(loss.item())
                all_recalls.append(recall)
            
            val_loss = np.mean(val_loss_items)
            val_recall_k = np.mean(all_recalls)
            best_val_loss = min(best_val_loss, val_loss)

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
                f"[Val] recall@10 : {val_recall_k:4.4%}, loss: {val_loss:4.2} || "
                f"best recall@10 : {best_val_recall_k:4.4%}, best loss: {best_val_loss:4.2} || "
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
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--dataset', type=str, default='GRU4RECDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=50, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='GRU4REC', help='model type (default: GRU4REC)')
    parser.add_argument('--optimizer', type=str, default='Adagrad', help='optimizer type (default: Adagrad)')
    parser.add_argument('--criterion', type=str, default='top1_max', help='criterion type (default: top1_max)')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (default: 1e-2)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 10)')
    parser.add_argument('--log_interval', type=int, default=1000, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping (default: 5)')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='batch accumulation step (default: 1)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/movie-recommendation/data/train/'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)