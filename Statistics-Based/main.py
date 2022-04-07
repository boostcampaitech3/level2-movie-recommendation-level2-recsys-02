import argparse
import pandas as pd
from preprocessing import *
from staticsmodel import *
from inference import *
import pickle

def main(args) :
    df = pd.read_csv(args.data_path + 'train_ratings.csv')
    df, movie_to_id, id_to_movie, popular_items = preprocess(df)
    
    model = StaticsBasedModel(df) 
    model.count_appearance()
    model.cal_ratio()
    
    with open('./st_model.pkl', 'wb') as f :
        pickle.dump(model, f)

    sub_u, sub_i = recommend(model, id_to_movie, popular_items)
    save_submission(sub_u, sub_i, args.save_path)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='../data/train/')
    parser.add_argument('--save_path', type=str, default='./')
    
    args = parser.parse_args()
    print(args)
    main(args)