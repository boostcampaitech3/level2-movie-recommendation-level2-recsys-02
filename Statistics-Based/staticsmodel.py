import torch
from tqdm import tqdm

class StaticsBasedModel() :
    def __init__(self, data) :
        item_num = len(data['item'].unique())
        self.appearance_ratio = torch.zeros((item_num, item_num))
        self.data = data
        self.users = data['user'].unique()

    def count_appearance(self) :
        print("count appearance ratio..")
        for user in tqdm(self.users) :
            user_item = self.data[self.data.user == user]['item'].values
            
            prev_item = user_item[0]
            for index, item in enumerate(user_item[1:-1], start=1) :
                next_item = user_item[index+1]
                self.appearance_ratio[item][prev_item] += 1
                self.appearance_ratio[item][next_item] += 1
                prev_item = item

    def cal_ratio(self) :
        appearance_sum = torch.sum(self.appearance_ratio, axis = 1)
        for item_index in range(1, appearance_sum.size()[0]) :
            self.appearance_ratio[item_index] = torch.div(self.appearance_ratio[item_index], appearance_sum[item_index])