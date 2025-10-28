import torch
from torch_geometric.data import Data

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备数量: {torch.cuda.device_count()}")
print(f"设备名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")

# 创建一个简单的图结构
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

print("\n图数据创建成功:")
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")

# 测试简单GNN模型
from torch_geometric.nn import GCNConv


class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


model = SimpleGNN()
print("\n模型构建成功:")
print(model)