import torch
from tqdm import tqdm

class StaticsBasedModel() :
    def __init__(self, data) :
        item_num = len(data['item'].unique())
        self.appearance_ratio = torch.zeros((item_num, item_num))
        self.appearance_ratio_a0 = None
        self.appearance_ratio_a1 = None
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
        self.appearance_ratio_a0, self.appearance_ratio_a1 = self.appearance_ratio.clone(), self.appearance_ratio.clone()

    def cal_ratio(self) :
        appearance_sum_a0 = torch.sum(self.appearance_ratio, axis = 0)
        appearance_sum_a1 = torch.sum(self.appearance_ratio, axis = 1)

        for item_index in range(appearance_sum_a0.size()[0]) :
            self.appearance_ratio_a0[:, item_index] = torch.div(self.appearance_ratio_a0[:, item_index], appearance_sum_a0[item_index])
            self.appearance_ratio_a1[item_index] = torch.div(self.appearance_ratio_a1[item_index], appearance_sum_a1[item_index])

        self.appearance_ratio = self.appearance_ratio_a0 * 0.3 + self.appearance_ratio_a1 * 0.7