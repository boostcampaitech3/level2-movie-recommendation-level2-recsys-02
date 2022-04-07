import torch
import pandas as pd

def get_rating(all_users, items_emb, user) :
    user_emb = all_users[user]
    rating = model.f(torch.matmul(user_emb, items_emb.t()))
    return rating


def inference(model, train_graph, user_to_id, file_name) :
    raw_df = pd.read_csv(data_path + 'train_ratings.csv')
    raw_df.head()
    
    popular_items = list(raw_df['item'].value_counts()[:700].keys())
    
    sub_u, sub_i = [], []
    model.eval()
    with torch.no_grad() :
        all_users_items = model(train_graph.to(device), model.embedding_user_item.weight.clone().to(device))
        all_users = all_users_items[:config["n_users"]]
        items_emb = all_users_items[config["n_users"]:]

        for user in tqdm(user_to_id.keys()) :
            user_idx = user_to_id[user]
            seen_list = list(raw_df[raw_df.user == user]['item'])

            rating = inference(all_users, items_emb, user_idx)
            top_ratings, top_items = torch.topk(torch.tensor(rating), len(rating))
            pred = []
            for idx in top_items :
                movie = id_to_movie[config["n_users"] + int(idx)]
                if movie in seen_list :
                    continue
                elif movie not in popular_items :
                    continue
                else : 
                    pred.append(movie) 
                    sub_u.append(user)
                    sub_i.append(movie)
                if len(pred) == 10 :
                    break

    submission = {"user" : sub_u, "item" : sub_i}
    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(f'./{file_name}.csv')