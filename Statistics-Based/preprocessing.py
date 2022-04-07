def preprocess(df) :
    print("preprocessing..")
    df = df.sort_values(['user', 'time'], ascending = [True, True])

    movies = df['item'].unique()
    movie_to_id = dict(zip(movies, range(len(movies))))
    id_to_movie = {v: k for k, v in movie_to_id.items()}
    
    df['item'] = df['item'].apply(lambda x : movie_to_id[x])
    popular_items = list(df['item'].value_counts()[:1000].keys())
    print("Complete!")
    return df, movie_to_id, id_to_movie, popular_items