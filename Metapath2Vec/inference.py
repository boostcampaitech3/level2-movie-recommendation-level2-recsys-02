from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
import os
from tqdm import tqdm

def inference(args, m2v, user_idx2id) :
    emb = m2v.skip_gram_model.u_embeddings.weight.cpu().data.numpy()
    
    cosine_sim = cosine_similarity(emb, emb)
    raw_rating_df = pd.read_csv(os.path.join(args.path, 'train_ratings.csv'))
    
    sub_u, sub_i = [], []
    for target_u in tqdm(range(0, 31360)) :
        target_u = int(user_idx2id[target_u][1:])
        target_seen = raw_rating_df[raw_rating_df.user == target_u]['item'].unique()

        target_seen_idx = []
        for seen in target_seen :
            target_seen_idx.append(m2v.data.word2id[f'i{seen}'])
        sim_score = torch.sum(torch.tensor(cosine_sim[target_seen_idx]), dim = 0)
        sim_score = torch.topk(torch.tensor(sim_score), len(sim_score))[1]

        cnt = 0
        for best_item in sim_score :
            item_id = int(m2v.data.id2word[int(best_item)][1:])
            if item_id in target_seen :
                continue
            sub_u.append(target_u)
            sub_i.append(item_id)
            cnt += 1
            if cnt == 10 :
                break
    
    ## save submission file
    submission = {"user" : sub_u, "item" : sub_i}
    
    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(args.submission_file)
    
    print("inference complete")