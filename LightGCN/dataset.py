from sklearn.metrics.pairwise import cosine_similarity

import pickle

def load_m2v_embeddings() :
    with open('../../Metapath2Vec/embeddings/m2v_item_emb.pkl', 'rb') as f :
        m2v_item_emb = pickle.load(f)
        
    with open('../../Metapath2Vec/embeddings/m2v_item_index.pkl', 'rb') as f :
        m2v_item_index = pickle.load(f)
             
    m2v_item_index_inverse = {v: k for k, v in m2v_item_index.items()}
    cosine_sim = torch.tensor(cosine_similarity(m2v_item_emb, m2v_item_emb))
    
    return cosine_sim, m2v_item_index, m2v_item_index_inverse

def load_neg_items() :
    with open('./neg_items.pkl', 'rb') as f :
        neg_items = pickle.load(f)
    return neg_items

# negative sampling with m2v embedding cosine similarity
class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, data, neg_items, cosine_sim, id_to_movie, movie_to_id, m2v_item_index, m2v_item_index_inverse):
        super(MovieDataset, self).__init__()
        self.users = data[:, 0]
        self.pos = data[:, 1]
        self.unseen = neg_items
        self.neg = self.get_neg_item(cosine_sim, id_to_movie, movie_to_id, m2v_item_index, m2v_item_index_inverse)
        
    def get_neg_item(self, cosine_sim, id_to_movie, movie_to_id, m2v_item_index, m2v_item_index_inverse) :
        neg = []
        for u_idx, p in tqdm(enumerate(self.pos)) :
            raw_p = id_to_movie[p] # raw movie id
            m2v_idx = m2v_item_index[f'i{raw_p}'] # for m2v index
            neg_idx = torch.topk(torch.tensor(cosine_sim[m2v_idx]), 100, largest= False)[1]
            # m2v index -> raw id -> lightgcn idx
            while True :
                rand_idx = np.random.choice(100, 1)[0]
                n = neg_idx[rand_idx]
                neg_item = movie_to_id[int(m2v_item_index_inverse[int(n)][1:])]
                if neg_item in self.unseen[self.users[u_idx]] :
                    neg.append(neg_item)
                    break
        return np.array(neg)
    
    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        #print(idx, self.users[idx], self.pos[idx], self.get_neg_item(users[idx]))
        return self.users[idx], self.pos[idx], self.neg[idx]
    