import os
import numpy as np
import pandas as pd
from tqdm import tqdm

data_dir = '/opt/ml/movie-recommendation/data/train/'

# 1. Rating df 생성
rating_data = data_dir + "/train_ratings.csv"
raw_rating_df = pd.read_csv(rating_data)

raw_rating_df['rating'] = 1.0 # implicit feedback
raw_rating_df.drop(['time'],axis=1,inplace=True)

users = set(raw_rating_df.loc[:, 'user'])
items = set(raw_rating_df.loc[:, 'item'])

#2. Genre df 생성
genre_data = data_dir + "genres.tsv"

raw_genre_df = pd.read_csv(genre_data, sep='\t')
raw_genre_df = raw_genre_df.drop_duplicates(subset=['item']) #item별 하나의 장르만 남도록 drop

genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df['genre']))}
raw_genre_df['genre']  = raw_genre_df['genre'].map(lambda x : genre_dict[x]) #genre id로 변경

# 3. Negative instance 생성
num_negative = 50
user_group_dfs = list(raw_rating_df.groupby('user')['item'])
first_row = True
user_neg_dfs = pd.DataFrame()

for u, u_items in user_group_dfs:
    u_items = set(u_items)
    i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)
    
    i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})
    if first_row == True:
        user_neg_dfs = i_user_neg_df
        first_row = False
    else:
        user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)

raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis = 0, sort=False)


# 4. Join dfs
joined_rating_df = pd.merge(raw_rating_df, raw_genre_df, left_on='item', right_on='item', how='inner')

# 5. user, item을 zero-based index로 mapping
users = list(set(joined_rating_df.loc[:,'user']))
users.sort()
items =  list(set((joined_rating_df.loc[:, 'item'])))
items.sort()
genres =  list(set((joined_rating_df.loc[:, 'genre'])))
genres.sort()

if len(users)-1 != max(users):
    users_dict = {users[i]: i for i in range(len(users))}
    joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
    users = list(set(joined_rating_df.loc[:,'user']))
    
if len(items)-1 != max(items):
    items_dict = {items[i]: i for i in range(len(items))}
    joined_rating_df['item']  = joined_rating_df['item'].map(lambda x : items_dict[x])
    items =  list(set((joined_rating_df.loc[:, 'item'])))

joined_rating_df = joined_rating_df.sort_values(by=['user'])
joined_rating_df.reset_index(drop=True, inplace=True)

data = joined_rating_df

data_dir = data_dir + 'context-aware/'
if not os.path.exists(data_dir) :
    os.mkdir(data_dir)
data.to_csv(os.path.join(data_dir, 'ratings_with_genres.csv'), mode='w', index=False)