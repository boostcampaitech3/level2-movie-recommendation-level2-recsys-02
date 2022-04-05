import torch
import torch.nn as nn
import os
from importlib import import_module


def load_model(saved_model, model_name, device=torch.device("cuda")):
    model_module = getattr(import_module("model"), model_name)
    model = model_module(
        input_dims=[31360,6807,12,18],
        embedding_dim=10
    )
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    for param in model.parameters():
        param.requires_grad=False

    return model


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
    def __init__(self, input_dim, embedding_dim, field_num, offsets):
        super(EmbeddingLayer, self).__init__()

        self.field_num = field_num
        self.offsets = torch.tensor(offsets, device='cuda')
        self.embedding = nn.Embedding(input_dim+1, embedding_dim, padding_idx=self.offsets[-1])

    def forward(self, x):
        one_hot_x = x[:,:self.field_num-1]
        multi_hot_x = x[:,self.field_num-1:].clone()

        embed_x = self.embedding(one_hot_x + self.offsets[:-1])

        sum_embed = []

        indices = multi_hot_x.nonzero()
        multi_hot_x[indices[:,0], indices[:,1]] = indices[:,1]+1
        embed = self.embedding(multi_hot_x + self.offsets[-1])
        sum_embed = torch.sum(embed, axis=1)

        embed_x= torch.cat([embed_x, sum_embed.unsqueeze(1)], axis=1)

        return embed_x


class ContextualBPR(BPR):
    def __init__(self, user_num, item_num, factor_num, context_dims = [12,30]):
        super(ContextualBPR, self).__init__(user_num,item_num,factor_num)

        field_num = len(context_dims)
        context_num = int(sum(context_dims))
        offsets = [0]+context_dims[:-1]

        self.total_embed_num = factor_num * field_num

        self.bias_item = nn.Parameter(torch.zeros(item_num))
        self.context_bias = EmbeddingLayer(context_num, 1, field_num, offsets)

        self.embed_context = EmbeddingLayer(context_num, factor_num, field_num, offsets)
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


class ContextualBPRv2(BPR):
    def __init__(self, user_num, item_num, factor_num, context_dim=10):
        super(ContextualBPRv2, self).__init__(user_num, item_num, factor_num)

        self.fm = load_model('/opt/ml/movie-recommendation/BPR/model/FM/', 'FM')
        for _ in filter(lambda p: p.requires_grad, self.fm.parameters()) :
            assert 'Freezing doesn\'t work'

        self.bias_item = nn.Parameter(torch.zeros(item_num))
        self.context_bias = nn.Linear(context_dim, 1, bias=False)

        self.embed_context = nn.Linear(context_dim, factor_num, bias=False)
        self.embed_user_context = nn.Embedding(user_num, factor_num)
    
    def forward(self, user, item_i, item_j, context_i, context_j):
        bpr_i, bpr_j = self.bpr(user, item_i, item_j)
        
        context_i = self.extract_features(torch.cat([user.unsqueeze(1), item_i.unsqueeze(1), context_i], dim=1))
        context_j = self.extract_features(torch.cat([user.unsqueeze(1), item_j.unsqueeze(1), context_j], dim=1))

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

        context_i = self.embed_context(context_i)
        context_j = self.embed_context(context_j)

        context_user = self.embed_user_context(user)

        prediction_i = (context_user * context_i).sum(dim=-1) + context_i_bias
        prediction_j = (context_user * context_j).sum(dim=-1) + context_j_bias
    
        return prediction_i, prediction_j
    
    def extract_features(self, x):
        embed_x = self.fm.embedding(x)
        bias = self.fm.bias + torch.sum(self.fm.fc(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2         
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        features = 0.5 * (square_of_sum - sum_of_square) + bias
        
        return features


class ContextualBPRv3(ContextualBPRv2):
    def __init__(self, user_num, item_num, factor_num, context_dim=20):
        super(ContextualBPRv3, self).__init__(user_num, item_num, factor_num)

        self.deepfm = load_model('/opt/ml/movie-recommendation/BPR/model/DeepFM/', 'DeepFM')
        for _ in filter(lambda p: p.requires_grad, self.deepfm.parameters()) :
            assert 'Freezing doesn\'t work'

        self.bias_item = nn.Parameter(torch.zeros(item_num))
        self.context_bias = nn.Linear(context_dim, 1, bias=False)

        self.embed_context = nn.Linear(context_dim, factor_num, bias=False)
        self.embed_user_context = nn.Embedding(user_num, factor_num)

        self.embedding_dim = self.deepfm.embedding_dim
    
    def forward(self, user, item_i, item_j, context_i, context_j):
        bpr_i, bpr_j = self.bpr(user, item_i, item_j)
        
        context_i = self.extract_features(torch.cat([user.unsqueeze(1), item_i.unsqueeze(1), context_i], dim=1))
        context_j = self.extract_features(torch.cat([user.unsqueeze(1), item_j.unsqueeze(1), context_j], dim=1))

        cbpr_i, cbpr_j = self.cbpr(user, context_i, context_j)

        return bpr_i + cbpr_i, bpr_j + cbpr_j
    
    def extract_features(self, x):
        embed_x = self.deepfm.embedding(x)

        bias = self.deepfm.bias + torch.sum(self.deepfm.fc(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2         
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        features = 0.5 * (square_of_sum - sum_of_square) + bias
        
        return torch.cat([features, self.deepfm.mlp_layers[:-1](embed_x.view(-1, self.embedding_dim))], dim=1)


class ContextualBPRv4(ContextualBPR):
    def __init__(self, user_num, item_num, factor_num, context_dim=128):
        super(ContextualBPRv4, self).__init__(user_num, item_num, factor_num)

        self.bias_item = nn.Parameter(torch.zeros(item_num))
        self.context_bias = nn.Linear(context_dim, 1, bias=False)

        self.embed_context = nn.Linear(context_dim, factor_num, bias=False)
        self.embed_user_context = nn.Embedding(user_num, factor_num)
    
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
        context_i_bias = self.context_bias(context_i).squeeze()
        context_j_bias = self.context_bias(context_j).squeeze()        

        context_i = self.embed_context(context_i)
        context_j = self.embed_context(context_j)

        context_user = self.embed_user_context(user)

        prediction_i = (context_user * context_i).sum(dim=-1) + context_i_bias
        prediction_j = (context_user * context_j).sum(dim=-1) + context_j_bias
    
        return prediction_i, prediction_j


class FM(nn.Module):
    def __init__(self, input_dims, embedding_dim):
        super(FM, self).__init__()
        self.field_num = len(input_dims)
        total_input_dim = int(sum(input_dims))
        self.offsets = [0]+input_dims[:-1]

        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = EmbeddingLayer(total_input_dim, 1, self.field_num, self.offsets)
        
        self.embedding = EmbeddingLayer(total_input_dim, embedding_dim, self.field_num, self.offsets)

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


class DeepFM(FM):
    def __init__(self, input_dims, embedding_dim, mlp_dims = [30, 20, 10], drop_rate=0.1):
        super(DeepFM, self).__init__(input_dims, embedding_dim)

        self.embedding_dim = embedding_dim * self.field_num

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


class GRU4REC(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=3, final_act='tanh',
                 dropout_hidden=0.5, dropout_input=0.5, batch_size=50, embedding_dim=-1, use_cuda=True):
        super(GRU4REC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = self.input_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.onehot_buffer = self.init_emb()
        self.h2o = nn.Linear(hidden_size, self.output_size)
        self.create_final_activation(final_act)
        if self.embedding_dim != -1:
            self.look_up = nn.Embedding(input_size, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        else:
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, input, hidden):
        '''
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.
        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''

        if self.embedding_dim == -1:
            embedded = self.onehot_encode(input)
            if self.training and self.dropout_input > 0: embedded = self.embedding_dropout(embedded)
            embedded = embedded.unsqueeze(0)
        else:
            embedded = input.unsqueeze(0)
            embedded = self.look_up(embedded)

        output, hidden = self.gru(embedded, hidden) #(num_layer, B, H)
        output = output.view(-1, output.size(-1))  #(B,H)
        logit = self.final_activation(self.h2o(output))

        return logit, hidden

    def init_emb(self):
        '''
        Initialize the one_hot embedding buffer, which will be used for producing the one-hot embeddings efficiently
        '''
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)
        return onehot_buffer

    def onehot_encode(self, input):
        """
        Returns a one-hot vector corresponding to the input
        Args:
            input (B,): torch.LongTensor of item indices
            buffer (B,output_size): buffer that stores the one-hot vector
        Returns:
            one_hot (B,C): torch.FloatTensor of one-hot vectors
        """
        self.onehot_buffer.zero_()
        index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(1, index, 1)
        return one_hot

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)
        mask = mask.to(self.device)
        input = input * mask
        return input

    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        try:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0