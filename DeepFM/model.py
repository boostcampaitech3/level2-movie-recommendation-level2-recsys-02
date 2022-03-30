import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

class DeepFM(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        self.field_num = len(input_dims)
        total_input_dim = int(sum(input_dims)) # n_user + n_movie + n_year + n_genre
        self.offset = int(sum(input_dims[:-1]))

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim+1, 1, padding_idx=self.offset)
        
        self.embedding = nn.Embedding(total_input_dim+1, embedding_dim, padding_idx=self.offset)
        self.embedding_dim = self.field_num * embedding_dim

        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i==0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim)) 
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x, embed_x):
        fm_y = self.bias + torch.sum(self.fc_layer(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2         
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y
    
    def mlp(self, x):
        inputs = x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def embedding_layer(self, x):
        one_hot_x = x[:,:self.field_num-1]
        multi_hot_x = x[:,self.field_num-1:].clone()

        embed_x = self.embedding(one_hot_x)

        sum_embed = []

        indices = multi_hot_x.nonzero()
        multi_hot_x[indices[:,0], indices[:,1]] = indices[:,1]+1
        embed = self.embedding(multi_hot_x + self.offset)
        sum_embed = torch.sum(embed, axis=1)

        embed_x= torch.cat([embed_x, sum_embed.unsqueeze(1)], axis=1)

        return embed_x
    
    def fc_layer(self, x):
        one_hot_x = x[:,:self.field_num-1]
        multi_hot_x = x[:,self.field_num-1:].clone()

        embed_x = self.fc(one_hot_x)
        
        indices = multi_hot_x.nonzero()
        multi_hot_x[indices[:,0], indices[:,1]] = indices[:,1]+1
        embed = self.fc(multi_hot_x + self.offset)
        sum_embed = torch.sum(embed, axis=1)

        embed_x= torch.cat([embed_x, sum_embed.unsqueeze(1)], axis=1)

        return embed_x

    def forward(self, x):
        #embedding component
        embed_x = self.embedding_layer(x)
        #fm component
        fm_y = self.fm(x, embed_x).squeeze(1)
        #deep component
        mlp_y = self.mlp(embed_x).squeeze(1)
        
        y = torch.sigmoid(fm_y + mlp_y)
        return y


class BPR(nn.Module):
	def __init__(self, user_num, item_num, factor_num):
		super(BPR, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""		
		self.embed_user = nn.Embedding(user_num, factor_num)
		self.embed_item = nn.Embedding(item_num, factor_num)

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

	def forward(self, user, item_i, item_j):
		user = self.embed_user(user)
		item_i = self.embed_item(item_i)
		item_j = self.embed_item(item_j)

		prediction_i = (user * item_i).sum(dim=-1)
		prediction_j = (user * item_j).sum(dim=-1)
		return prediction_i, prediction_j