# model
import torch.optim

from modules_with_edge_features import *

class MainModel(nn.Module):
    def __init__(self,dr,lr,nlayers,lamda,alpha,atten_time,nfeats):
        super(MainModel, self).__init__()

        self.drop1 = nn.Dropout(p=dr)
        self.fc1 = nn.Linear(640*atten_time, 256)  # for attention

        self.drop2 = nn.Dropout(p=dr)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

        self.rgn_egnn = RGN_EGNN(nlayers=2, nfeat=nfeats, nhidden=512, nclass=1, dropout=dr,
                                 lamda=lamda, alpha=alpha, variant=True, heads=1)
        self.rgn_gcn2 = RGN_GCN(nlayers=nlayers, nfeat=nfeats, nhidden=128, nclass=1,
                                dropout=dr,
                                lamda=lamda, alpha=alpha, variant=True, heads=1)

        self.multihead_attention = nn.ModuleList([Attention_1(hidden_size=512+128, num_attention_heads=16) for _ in range(atten_time)])  # gated attention  收敛比较快.本来是8

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay=1e-16)

    def forward(self, G, h, x,adj,efeats):

        h = torch.squeeze(h)
        x = torch.squeeze(x)
        h = h.to(torch.float32)

        fea1 = self.rgn_egnn(G, h, x,efeats)
        fea1 = torch.unsqueeze(fea1, dim=0)

        fea2 = self.rgn_gcn2(h, adj)
        fea2 = torch.unsqueeze(fea2, dim=0)

        fea = torch.cat([fea1,fea2],dim=2)

        # gated self-attention
        attention_outputs = []
        for i in range(len(self.multihead_attention)):
            multihead_output, _ = self.multihead_attention[i](fea)
            attention_outputs.append(multihead_output)
        embeddings = torch.cat(attention_outputs, dim=2)

        # # self-attention
        # attention_outputs = []
        # for i in range(len(self.multihead_attention)):
        #     multihead_output, _ = self.multihead_attention[i](fea,fea,fea)
        #     attention_outputs.append(multihead_output)
        # embeddings = torch.cat(attention_outputs, dim=2)

        out = self.drop1(embeddings)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)

        out = self.fc2(out)

        return out

