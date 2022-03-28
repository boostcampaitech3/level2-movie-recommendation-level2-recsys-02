import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

class DeepFM(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        self.field_num = len(input_dims)
        total_input_dim = int(sum(input_dims)) # n_user + n_movie + n_genre
        self.offset = total_input_dim - 18

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim, 1)
        
        self.embedding = nn.Embedding(total_input_dim, embedding_dim) 
        self.embedding_dim = self.field_num * embedding_dim

        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i==0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim)) #TODO 1 : linear layer를 넣어주세요.
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x, embed_x):
        fm_y = self.bias + torch.sum(self.fc_layer(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2         #TODO 2 : torch.sum을 이용하여 square_of_sum을 작성해주세요(hint : equation (2))
        sum_of_square = torch.sum(embed_x ** 2, dim=1)         #TODO 3 : torch.sum을 이용하여 sum_of_square을 작성해주세요(hint : equation (2))
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y
    
    def mlp(self, x):
        inputs = x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def embedding_layer(self, x):
        one_hot_x = x[:,:self.field_num-1]
        multi_hot_x = x[:,self.field_num-1:]

        embed_x = self.embedding(one_hot_x)

        sum_embed = []

        for mhx in multi_hot_x :
            genres = torch.where(mhx)
            embed_genres = self.embedding(genres[0] + self.offset)
            sum_embed.append(torch.sum(embed_genres, axis=0).unsqueeze(0))
        sum_embed = torch.cat(sum_embed, axis=0)

        embed_x= torch.cat([embed_x, sum_embed.unsqueeze(1)], axis=1)

        return embed_x
    
    def fc_layer(self, x):
        one_hot_x = x[:,:self.field_num-1]
        multi_hot_x = x[:,self.field_num-1:]

        embed_x = self.fc(one_hot_x)

        sum_embed = []

        for mhx in multi_hot_x :
            genres = torch.where(mhx)
            embed_genres = self.fc(genres[0] + self.offset)
            sum_embed.append(torch.sum(embed_genres, axis=0).unsqueeze(0))
        sum_embed = torch.cat(sum_embed, axis=0)

        embed_x= torch.cat([embed_x, sum_embed.unsqueeze(1)], axis=1)

        return embed_x

    def forward(self, x):
        #embedding component
        # start = time.time()
        embed_x = self.embedding_layer(x)
        #fm component
        fm_y = self.fm(x, embed_x).squeeze(1)
        #deep component
        mlp_y = self.mlp(embed_x).squeeze(1)
        
        y = torch.sigmoid(fm_y + mlp_y)
        # print(time.time() - start)
        return y