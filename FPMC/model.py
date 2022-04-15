import torch
import torch.nn as nn

class FPMC(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(FPMC, self).__init__()	
        self.embed_UI = nn.Embedding(user_num, factor_num)
        self.embed_IU = nn.Embedding(item_num, factor_num)
        self.embed_LI = nn.Embedding(item_num+1, factor_num, padding_idx=0)
        self.embed_IL = nn.Embedding(item_num, factor_num)
        
        nn.init.normal_(self.embed_UI.weight, std=0.01)
        nn.init.normal_(self.embed_IU.weight, std=0.01)
        nn.init.normal_(self.embed_LI.weight, std=0.01)
        nn.init.normal_(self.embed_IL.weight, std=0.01)
        
    def forward(self, user, item, item_seq, seq_len):
        VUI = self.embed_UI(user) # (batch_size, factor_num)
        VIU = self.embed_IU(item) # (batch_size, factor_num)
        VLI = self.embed_LI(item_seq) # (batch_size, sequence_len, factor_num)
        VIL = self.embed_IL(item) # (batch_size, factor_num)

        VUI_m_VIU = torch.sum(VUI*VIU, axis=1)
        VLI_m_VIL = torch.sum(torch.bmm(VLI, (VIL.unsqueeze(2))), axis=1) / seq_len.unsqueeze(1)

        return VUI_m_VIU + VLI_m_VIL.squeeze()
    
    def predict(self, user, items, item_seq, seq_len):
        VUI = self.embed_UI(user) # (batch_size, factor_num)
        VIU = self.embed_IU(items) # (batch_size, item_num, factor_num)
        VLI = self.embed_LI(item_seq) # (batch_size, sequence_len, factor_num)
        VIL = self.embed_IL(items) # (batch_size, item_num, factor_num)

        VUI_m_VIU = torch.bmm(VIU, VUI.unsqueeze(2))
        VLI_m_VIL = torch.sum(torch.bmm(VIL, VLI.transpose(1,2)), axis=2) / seq_len.unsqueeze(1)

        return VUI_m_VIU.squeeze() + VLI_m_VIL