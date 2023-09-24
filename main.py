import torch
import networkx as nx
from torch_geometric.datasets import KarateClub
import time
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.nn import GCNConv

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

model = GCN()
print(model)

# train
model = GCN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(data):
    model.train()
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)    # h是2维向量，方便画图
    loss = criterion(out[data.train_mask], data.y[data.train_mask])    # semi-supervised
    label = dataset.y
    _, indx = out.max(dim=-1)
    # print(label,  indx, dataset.train_mask)
    un_mask = label[dataset.train_mask]==indx[dataset.train_mask]
    print(f'acc:%.4f, loss:%.4f' % (sum(un_mask)/len(un_mask), loss))
    loss.backward()
    optimizer.step()
    return loss, h

for epoch in range(100):
    loss, h = train(data=dataset)
    # print(f'loss:{loss}')