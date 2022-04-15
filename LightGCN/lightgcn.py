import dgl
import torch
import torch.nn as nn

def getEmbedding(model, user, pos, neg, output) :
    userEmb0 = model.embedding_user_item(user)
    posEmb0 = model.embedding_user_item(pos)
    negEmb0 = model.embedding_user_item(neg)
    
    user_emb = output[user]
    pos_emb = output[pos]
    neg_emb = output[neg]
    
    return userEmb0, posEmb0, negEmb0, user_emb, pos_emb, neg_emb

def bpr_loss(model, users, pos, neg, data, device):
    # assuming we always sample the same number of positive and negative sample
    # per user
    (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0) = getEmbedding(model, users.long().to(device), pos.long().to(device), neg.long().to(device), data.to(device))
    reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                        posEmb0.norm(2).pow(2)  +
                        negEmb0.norm(2).pow(2))/float(len(users))
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)
    
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    
    return loss, reg_loss

class LightGCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(LightGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.gconv = dgl.nn.pytorch.conv.GraphConv(self.in_channels, self.out_channels, norm='both', weight=False, bias = False)

    def forward(self, g, emb):
        out = self.gconv(g, emb)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class LightGCN(nn.Module):
    def __init__(self, num_user, num_item, args, device=None):
        super().__init__()
        self.num_users  = num_user
        self.num_items  = num_item
        self.embedding_size = args.embedding_size
        self.in_channels = self.embedding_size
        self.out_channels = self.embedding_size
        self.num_layers = args.num_layers

        # 0-th layer embedding.
        self.embedding_user_item = torch.nn.Embedding(
            num_embeddings=self.num_users + self.num_items,
            embedding_dim=self.embedding_size)

        self.alpha = None

        nn.init.normal_(self.embedding_user_item.weight, std=0.1)
        
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(
                LightGCNConv(self.embedding_size, self.embedding_size))
        self.f = nn.Sigmoid()
        
        self.device = None
        if device is not None:
            self.convs.to(device)
            self.device = device
    
    def forward(self, g, emb):
        emb_lst = []
        for i in range(self.num_layers):
            x = self.convs[i](g, emb)
            if self.device is not None:
                x = x.to(self.device)
            emb_lst.append(x)
        
        emb_lst = torch.stack(emb_lst)
        
        self.alpha = []
        for k in range(self.num_layers) :
            alpha = 1 / (1 + k) * torch.ones(emb_lst.shape[1:])
            self.alpha.append(alpha)
        self.alpha = torch.stack(self.alpha)

        if self.device is not None:
            self.alpha = self.alpha.to(self.device)
            emb_lst = emb_lst.to(self.device)

        x = (emb_lst * self.alpha).sum(dim=0)  # Sum along K layers.
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
    
    
    
    