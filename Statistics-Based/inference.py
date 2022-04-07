import torch
import pandas as pd
from tqdm import tqdm

def recommend(model, id_to_movie, popular_items) :
    sub_u, sub_i = [], []
    print("inference...")
    for user in tqdm(model.users) :
        seen_item = torch.tensor(model.data[model.data.user == user]['item'].values)
        score = torch.sum(model.appearance_ratio[seen_item], axis = 0)
        ranking = torch.topk(score, len(score))[1]

        pred = set()
        for item_id in ranking :
            if item_id in seen_item :
                continue
            if item_id not in popular_items :
                continue
            movie = id_to_movie[int(item_id)]
            sub_u.append(user)
            sub_i.append(movie)
            pred.add(movie)
            if len(pred) == 10 :
                break
    return sub_u, sub_i

def save_submission(sub_u, sub_i, save_path) :
    submission = {"user" : sub_u, "item" : sub_i}
    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(f'{save_path}statics_submission.csv')