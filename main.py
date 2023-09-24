import torch
import networkx as nx
from torch_geometric.datasets import KarateClub
import time
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

dataset = KarateClub()    # data
class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)   # 输入特征维度， 输出特征维度
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.cls = nn.Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        # 输出
        h = h.tanh()

        # cls
        out = self.cls(h)
        return out, h

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.lin1 = nn.Linear(dataset.num_features, 4)   # 输入特征维度， 输出特征维度
        self.lin2 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model_gcn = GCN()
print(model_gcn)

# train
criterion_gcn = nn.CrossEntropyLoss()
optimizer_gcn = torch.optim.Adam(model_gcn.parameters(), lr=0.01)

def train_GCN(data):
    model_gcn.train()
    optimizer_gcn.zero_grad()
    out, h = model_gcn(data.x, data.edge_index)    # h是2维向量，方便画图
    loss = criterion_gcn(out[data.train_mask], data.y[data.train_mask])    # semi-supervised
    label = dataset.y
    _, indx = out.max(dim=-1)
    # print(label,  indx, dataset.train_mask)
    un_mask = label[dataset.train_mask]==indx[dataset.train_mask]    # 只关注un-mask位置是否相同
    # print(f'acc:%.4f, loss:%.4f' % (sum(un_mask)/len(un_mask), loss))
    loss.backward()
    optimizer_gcn.step()
    return sum(un_mask)/len(un_mask), loss

model_mlp = MLP()
print(model_mlp)

# train
criterion_mlp = nn.CrossEntropyLoss()
optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=0.01)

def train_MLP(data):
    model_mlp.train()
    optimizer_mlp.zero_grad()
    out= model_mlp(data.x)    # h是2维向量，方便画图
    loss = criterion_mlp(out[data.train_mask], data.y[data.train_mask])    # semi-supervised
    label = dataset.y
    _, indx = out.max(dim=-1)
    # print(label,  indx, dataset.train_mask)
    un_mask = label[dataset.train_mask]==indx[dataset.train_mask]    # 只关注un-mask位置是否相同
    # print(f'acc:%.4f, loss:%.4f' % (sum(un_mask)/len(un_mask), loss))
    loss.backward()
    optimizer_mlp.step()
    return sum(un_mask)/len(un_mask), loss

for epoch in range(100):
    acc1, loss1 = train_GCN(data=dataset)
    acc2, loss2 = train_MLP(data=dataset)
    print(f'GCN.loss:{loss1}, GCN.acc:{acc1}, MLP.loss:{loss2}, MLP.acc:{acc2}')