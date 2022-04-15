import argparse
import glob
import os
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from dataset import VAEDataLoader
from loss import loss_function_vae
import bottleneck as bn
import time
import json


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    tmp2 = np.minimum(k, X_true_binary.sum(axis=1))

    recall = tmp / tmp2
    return recall


def load_model(model_path, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()

    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    np.random.seed(98765)
    
    for _, group in data_grouped_by_user:
        n_items_u = len(group)
        
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        
        else:
            tr_list.append(group)
    
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def train(model, criterion, optimizer, train_data, device, epoch, is_VAE = False):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    N = train_data.shape[0]
    idxlist = list(range(N))

    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(device)
        optimizer.zero_grad()

        if is_VAE:
          if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 1.*update_count / args.total_anneal_steps)
          else:
              anneal = args.anneal_cap

          optimizer.zero_grad()
          recon_batch, mu, logvar = model(data)
          
          loss = criterion(recon_batch, data, mu, logvar, anneal)
        else:
          recon_batch = model(data)
          loss = criterion(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))
            
            start_time = time.time()
            train_loss = 0.0


def evaluate(model, criterion, train_data, data_tr, data_te, device, is_VAE=False):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    N = train_data.shape[0]
    n100_list = []
    r10_list = []
    r20_list = []
    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)
            if is_VAE :
              
              if args.total_anneal_steps > 0:
                  anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
              else:
                  anneal = args.anneal_cap

              recon_batch, mu, logvar = model(data_tensor)

              loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

            else :
              recon_batch = model(data_tensor)
              loss = criterion(recon_batch, data_tensor)

            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r10 = Recall_at_k_batch(recon_batch, heldout_data, 10)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            

            n100_list.append(n100)
            r10_list.append(r10)
            r20_list.append(r20)
 
    total_loss /= len(range(0, e_N, args.batch_size))
    n100_list = np.concatenate(n100_list)
    r10_list = np.concatenate(r10_list)
    r20_list = np.concatenate(r20_list)

    return total_loss, np.mean(n100_list), np.mean(r10_list), np.mean(r20_list)
    

def k_fold(data_dir, model_dir, output_dir, args) :
    '''
    cutmix와 mixup을 통해 data augmentation을 하며 모델을 학습하며 
    k개의 모델을 학습해 최종 결과를 도출하는 k-fold ensemble을 진행합니다.
    :param data_dir: data가 담긴 directory
    :param model_dir: 학습된 model이 담길 directory
    :param output_dir: 최종 예측 결과가 담길 directory
    :param args: 모델의 hyperparameter들을 담은 arguments
    '''
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # -- settings
    global update_count
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    raw_data = pd.read_csv(os.path.join(data_dir, 'train_ratings.csv'), header=0)
    raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=5, min_sc=0)

    unique_uid = user_activity.index
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    unique_sid = pd.unique(raw_data['item'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(data_dir+f'{args.dropout}', 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_uid.txt'), 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    loader = VAEDataLoader(data_dir+f'{args.dropout}')
    n_items = loader.load_n_items()

    test_plays = raw_data.loc[raw_data['user'].isin(unique_uid)]
    test_data = numerize(test_plays, profile2id, show2id)
    test_data.to_csv(os.path.join(pro_dir, 'test.csv'), index=False, mode='w')

    kf = KFold(n_splits=args.n_splits)
    patience = args.patience

    # K-Fold Cross Validation과 동일하게 Train, Valid Index를 생성합니다. 
    for i, (train_idx, valid_idx) in enumerate(kf.split(np.arange(unique_uid.size))):
        print('Loading Dataset ...', end=' ')

        tr_users = unique_uid[train_idx]
        vd_users = unique_uid[valid_idx]
        
        train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]

        train_data = numerize(train_plays, profile2id, show2id)
        train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False, mode='w')

        vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
        vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_sid)]
        vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

        vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
        vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False, mode='w')

        vad_data_te = numerize(vad_plays_te, profile2id, show2id)
        vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False, mode='w')

        train_data = loader.load_data('train')
        vad_data_tr, vad_data_te = loader.load_data('validation')

        print("Done!")

        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            p_dims = [600, 1200, n_items],
            dropout=args.dropout
        ).to(device)

        # -- loss & metric
        criterion = loss_function_vae
        opt_module = getattr(import_module("torch.optim"), args.optimizer)
        optimizer = opt_module(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-5,
        )
        # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        counter = 0
        best_val_loss = np.inf
        update_count = 0

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(model, criterion, optimizer, train_data, device, epoch, is_VAE = True)
            val_loss, n100, r10, r20 = evaluate(model, criterion, train_data, vad_data_tr, vad_data_te, device, is_VAE=True)
            print('| {:2d} fold epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                    'n100 {:5.3f} | r10 {:5.3f} | r20 {:5.3f}'.format(
                        i+1, epoch, time.time() - epoch_start_time, val_loss,
                        n100, r10, r20), f" | counter/patience : {counter}/{patience}")

            # Save the model if the r10 is the best we've seen so far.
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f"{save_dir}/cv{i+1}.pth")
                best_val_loss = val_loss
                counter = 0
            else :
                counter += 1
            
            if counter > patience:
                print("Early Stopping...")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1)')
    #parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--batch_size', type=int, default=500, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='MultiVAE', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    #parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    #parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--total_anneal_steps', type=int, default=200000, help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2, help='largest annealing parameter')
    parser.add_argument('--dropout', type=float, default=0.5, help='learning rate (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping (default: 5)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--n_splits', type=int, default=20, help='k for stratified k-fold')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/movie-recommendation/data/train/'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output/kfold'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    update_count = 0

    k_fold(data_dir, model_dir, output_dir, args)