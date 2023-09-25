# 创建torch_geometric的输入数据
import torch

x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)    # 输入数据
y = torch.tensor([0,1,0,1])    # 标签
# 边的顺序无所谓
edge_index = torch.tensor([[0,1,2,0,3],    # 起始点
                           [1,0,1,3,2]], dtype=torch.long)    # 终止点
# 创建torch_geometric中的图
from torch_geometric.data import Data
data = Data(x=x, y=y, edge_index=edge_index)
print(data)