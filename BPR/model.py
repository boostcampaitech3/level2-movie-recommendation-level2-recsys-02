import torch
import torch.nn as nn

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


class ContextualBPR(BPR):
    def __init__(self, user_num, item_num, factor_num, context_dims):
        super(ContextualBPR, self).__init__(user_num,item_num,factor_num)

        field_num = len(context_dims)
        context_num = int(sum(context_dims))
        offset = int(sum(context_dims[:-1]))

        self.total_embed_num = factor_num * field_num

        self.bias_item = nn.Parameter(torch.zeros(item_num))
        self.context_bias = EmbeddingLayer(context_num, 1, field_num, offset)

        self.embed_context = EmbeddingLayer(context_num, factor_num, field_num, offset)
        self.embed_user_context = nn.Embedding(user_num, self.total_embed_num)
    
    def forward(self, user, item_i, item_j, context_i, context_j):
        bpr_i, bpr_j = self.bpr(user, item_i, item_j)
        cbpr_i, cbpr_j = self.cbpr(user, context_i, context_j)

        return bpr_i + cbpr_i, bpr_j + cbpr_j
    
    def bpr(self, user, item_i, item_j):
        user = self.embed_user(user)
        b_i = self.bias_item[item_i]
        b_j = self.bias_item[item_j]
        
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)
        
        prediction_i = (user * item_i).sum(dim=-1) + b_i
        prediction_j = (user * item_j).sum(dim=-1) + b_j

        return prediction_i, prediction_j
    
    def cbpr(self, user, context_i, context_j):
        context_i_bias = torch.sum(self.context_bias(context_i), dim=1).squeeze()
        context_j_bias = torch.sum(self.context_bias(context_j), dim=1).squeeze()        

        context_i = self.embed_context(context_i).view(-1, self.total_embed_num)
        context_j = self.embed_context(context_j).view(-1, self.total_embed_num)

        context_user = self.embed_user_context(user)

        prediction_i = (context_user * context_i).sum(dim=-1) + context_i_bias
        prediction_j = (context_user * context_j).sum(dim=-1) + context_j_bias
    
        return prediction_i, prediction_j