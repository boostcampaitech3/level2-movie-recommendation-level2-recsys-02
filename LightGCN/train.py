import os
import argparse
from preprocess import *
from dataset import * 
from lightgcn import *
from inference import *

def train(args) :
    # preprocessing
    train_graph, id_to_movie, id_to_user, data_array_train, data_array_test, num_user, num_item = preprocess(args.data_path)
    cosine_sim, m2v_item_index, m2v_item_index_inverse = load_m2v_embeddings()
    neg_items = load_neg_items()
    
    train_dataset = MovieDataset(data_array_train, neg_items, cosine_sim, id_to_movie, movie_to_id, m2v_item_index, m2v_item_index_inverse)
    test_dataset = MovieDataset(data_array_test, neg_items, cosine_sim, id_to_movie, movie_to_id, m2v_item_index, m2v_item_index_inverse)
    
    train_loader = DataLoader(train_dataset, batch_size = 1024)
    test_loader = DataLoader(test_dataset, batch_size = 1024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LightGCN(num_user, num_item, args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)
    
    best_test_loss = int(1e9)
    cnt = 0
    for epoch in tqdm(range(100)):
        print("Training on the {} epoch".format(epoch))
        model.train()
        loss_sum = 0
        for user, pos, neg in train_loader:
            optimizer.zero_grad()
            output = model(train_graph.to(device), model.embedding_user_item.weight.clone().to(device))

            loss, reg_loss = bpr_loss(model, user, pos, neg, output, device)
            reg_loss = reg_loss * args.weight_decay
            loss = loss + reg_loss
            loss_sum += loss.detach()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        model.eval()
        test_loss_sum = 0
        with torch.no_grad() :
            for user, pos, neg in test_loader:
                output = model(train_graph.to(device), model.embedding_user_item.weight.clone().to(device))
                test_loss, test_reg_loss = bpr_loss(model, user, pos, neg, output, device)
                test_reg_loss = test_reg_loss * args.weight_decay
                test_loss = test_loss + test_reg_loss
                test_loss_sum += loss.detach()

        if best_test_loss > test_loss_sum :
            best_test_loss = test_loss_sum
            torch.save(model.state_dict(), f'./{args.file_name}.pt')
            cnt = 0
        else :
            cnt += 1
            if cnt == 10 :
                break
        print(f'Epoch {epoch} || train loss : {loss_sum}  |  test_loss {test_loss_sum}  | best_test_loss {best_test_loss}')

    # inference
    inference(model, train_graph, user_to_id, args.file_name)
    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--model', type=str, default='Metapath2Vec', help='model type (default: Metapath2Vec)')
    parser.add_argument('--data_path', type=str, default='../data/train')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--weight_decay', type=int, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--file_name', type=str)
    
    args = parser.parse_args()
    print(args)
    
    train(args)
    