import torch
import torch.nn as nn

class FM(nn.Module):
    def __init__(self, input_dims, embedding_dim):
        super(DeepFM, self).__init__()
        self.field_num = len(input_dims)
        total_input_dim = int(sum(input_dims))
        self.offset = int(sum(input_dims[:-1]))

        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = EmbeddingLayer(total_input_dim+1, 1, self.field_num, self.offset)
        
        self.embedding = EmbeddingLayer(total_input_dim+1, embedding_dim, self.field_num, self.offset)
        self.embedding_dim = self.field_num * embedding_dim

    def fm(self, x, embed_x):
        fm_y = self.bias + torch.sum(self.fc(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2         
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y

    def forward(self, x):
        #embedding component
        embed_x = self.embedding(x)
        #fm component
        fm_y = self.fm(x, embed_x).squeeze(1)

        y = torch.sigmoid(fm_y)
        return y


class DeepFM(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        self.field_num = len(input_dims)
        total_input_dim = int(sum(input_dims)) # n_user + n_movie + n_year + n_genre
        self.offset = int(sum(input_dims[:-1]))

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = EmbeddingLayer(total_input_dim+1, 1, self.field_num, self.offset)
        
        self.embedding = EmbeddingLayer(total_input_dim+1, embedding_dim, self.field_num, self.offset)
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
        fm_y = self.bias + torch.sum(self.fc(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2         
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y
    
    def mlp(self, x):
        inputs = x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        #embedding component
        embed_x = self.embedding(x)
        #fm component
        fm_y = self.fm(x, embed_x).squeeze(1)
        #deep component
        mlp_y = self.mlp(embed_x).squeeze(1)
        
        y = torch.sigmoid(fm_y + mlp_y)
        return y


class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, embedding_dim, field_num, offset):
        super(EmbeddingLayer, self).__init__()

        self.field_num = field_num
        self.offset = offset
        self.embedding = nn.Embedding(input_dim+1, embedding_dim, padding_idx=self.offset)

    def forward(self, x):
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