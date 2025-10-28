import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, degree
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import time

# 1. 加载数据集
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]


# 2. 计算最短路径距离（SPD）矩阵 - 添加剪裁
def compute_spd_matrix(edge_index, num_nodes, max_spd=10):
    # 转换为SciPy稀疏矩阵
    row, col = edge_index.cpu().numpy()
    adj = coo_matrix((np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes))

    # 计算最短路径距离
    dist_matrix = shortest_path(adj, directed=False, unweighted=True)

    # 处理未连接的节点对（设置为最大距离+1）
    dist_matrix[np.isinf(dist_matrix)] = max_spd + 1

    # 安全处理异常值
    dist_matrix = np.nan_to_num(dist_matrix, nan=max_spd+1)
    dist_matrix = np.clip(dist_matrix, 0, max_spd)

    # 转换为torch张量
    return torch.from_numpy(dist_matrix).float()


spd_matrix = compute_spd_matrix(data.edge_index, num_nodes=data.num_nodes, max_spd=10)


# 3. 修复节点中心性计算（无向图处理）
def compute_centrality_fixed(edge_index, num_nodes):
    # 统一计算度（无向图入度=出度）
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
    # 确保度数非负
    deg = torch.clamp(deg, min=0)

    return deg, deg


deg_in, deg_out = compute_centrality_fixed(data.edge_index, data.num_nodes)


# 4. 计算边编码矩阵 - 添加剪裁
def compute_edge_encoding(edge_index, num_nodes, spd_matrix, max_distance=5):
    """
    创建边编码矩阵：
    - 对于直接连接的节点(i,j)，使用边特征（这里用1表示存在边）
    - 对于非直接连接的节点，使用最短路径距离
    """
    # 创建邻接矩阵（1表示有边，0表示无边）
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]

    # 初始化边编码矩阵
    edge_encoding = torch.zeros(num_nodes, num_nodes, dtype=torch.long)

    # 直接边标记为1
    edge_encoding[adj > 0] = 1

    # 非直接边使用SPD距离（从2开始编码）
    edge_encoding[(adj == 0) & (spd_matrix > 1)] = torch.clamp(spd_matrix[(adj == 0) & (spd_matrix > 1)], 2,
                                                               max_distance).long()

    return edge_encoding


edge_encoding = compute_edge_encoding(data.edge_index, data.num_nodes, spd_matrix, max_distance=5)


# 5. 增强的GraphormerLayer - 添加注意力掩码和Dropout
class GraphormerLayer(nn.Module):
    def __init__(self, dim, heads=8, max_spd_value=10, max_centrality=50, max_edge_encoding=5, drop_rate=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.drop_rate = drop_rate

        # SPD嵌入层
        self.max_spd_value = max_spd_value
        self.spd_embedding = nn.Embedding(max_spd_value + 1, heads)  # 输出[heads]维的偏置

        # 节点中心性编码嵌入层
        self.max_centrality = max_centrality
        self.centrality_encoder_in = nn.Embedding(max_centrality + 1, dim)
        self.centrality_encoder_out = nn.Embedding(max_centrality + 1, dim)

        # 边编码嵌入层
        self.max_edge_encoding = max_edge_encoding
        self.edge_embedding = nn.Embedding(max_edge_encoding + 1, heads)  # 每个头一个标量

        # 注意力层
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # FFN层
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Dropout层
        self.attn_dropout = nn.Dropout(drop_rate)
        self.ffn_dropout = nn.Dropout(drop_rate)

    def forward(self, x, spd_matrix, centrality_in, centrality_out, edge_encoding):
        # 安全剪裁输入值
        centrality_in = torch.clamp(centrality_in, min=0, max=self.max_centrality)
        centrality_out = torch.clamp(centrality_out, min=0, max=self.max_centrality)
        edge_encoding = torch.clamp(edge_encoding, min=0, max=self.max_edge_encoding)

        # 残差连接
        residual = x

        # 1. 添加节点中心性编码
        centrality_emb = (self.centrality_encoder_in(centrality_in) +
                          self.centrality_encoder_out(centrality_out))
        x = x + centrality_emb

        # 2. 多头注意力
        q = self.q_proj(x).view(-1, self.heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.heads, self.head_dim)

        # 3. 注意力分数
        attn_scores = torch.einsum('ihd,jhd->ijh', q, k) / (self.head_dim ** 0.5)

        # 4. 添加SPD偏置
        spd_indices = torch.clamp(spd_matrix, 0, self.max_spd_value).long()
        spd_bias = self.spd_embedding(spd_indices)  # 形状 [N, N, heads]

        # 5. 添加边编码偏置
        edge_indices = torch.clamp(edge_encoding, 0, self.max_edge_encoding).long()
        edge_bias = self.edge_embedding(edge_indices)  # 形状 [N, N, heads]

        attn_scores = attn_scores + spd_bias + edge_bias

        # 关键修复：添加注意力掩码（屏蔽无效位置）
        valid_mask = (spd_matrix <= self.max_spd_value) & (edge_encoding <= self.max_edge_encoding)
        attn_scores = attn_scores.masked_fill(~valid_mask[..., None], -1e9)

        # 6. 归一化注意力权重
        attn_weights = F.softmax(attn_scores, dim=1)  # 在j维度上归一化

        # 添加注意力Dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 7. 注意力输出
        attn_output = torch.einsum('ijh,jhd->ihd', attn_weights, v)

        # 8. 拼接多头输出
        attn_output = attn_output.reshape(-1, self.dim)
        attn_output = self.out_proj(attn_output)

        # 9. 残差连接 & 层归一化
        x = self.norm1(residual + attn_output)

        # 10. FFN部分
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.ffn_dropout(ffn_output))

        return x


# 6. 完整的Graphormer模型 - 添加输入Dropout和层间Dropout
class Graphormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2,
                 max_centrality=50, max_spd_value=10, max_edge_encoding=5, drop_rate=0.5):
        super().__init__()
        # 输入嵌入层
        self.embed = nn.Linear(input_dim, hidden_dim)

        # 输入Dropout层
        self.input_dropout = nn.Dropout(drop_rate)

        # 中心性编码嵌入层（用于输入）
        self.max_centrality = max_centrality
        self.input_centrality_in = nn.Embedding(max_centrality + 1, hidden_dim)
        self.input_centrality_out = nn.Embedding(max_centrality + 1, hidden_dim)

        # Graphormer层
        self.layers = nn.ModuleList([
            GraphormerLayer(
                hidden_dim,
                max_spd_value=max_spd_value,
                max_centrality=max_centrality,
                max_edge_encoding=max_edge_encoding,
                drop_rate=0.1 if i == num_layers - 1 else 0.3  # 最后层用较小的dropout
            ) for i in range(num_layers)
        ])

        # 层间Dropout
        self.layer_dropout = nn.Dropout(0.2)

        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 初始化权重
        self.apply(self.init_weights)

        # 基于全局数据计算安全的max_centrality
        self.max_centrality = max_centrality
        self.max_spd_value = max_spd_value
        self.max_edge_encoding = max_edge_encoding

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, spd_matrix, centrality_in, centrality_out, edge_encoding):
        # 安全剪裁所有输入
        centrality_in = torch.clamp(centrality_in, min=0, max=self.max_centrality)
        centrality_out = torch.clamp(centrality_out, min=0, max=self.max_centrality)
        edge_encoding = torch.clamp(edge_encoding, min=0, max=self.max_edge_encoding)
        spd_matrix = torch.clamp(spd_matrix, min=0, max=self.max_spd_value)


        # 初始节点特征嵌入
        x = self.embed(x)
        x = self.input_dropout(x)

        # 添加输入级的中心性编码
        centrality_emb = (self.input_centrality_in(centrality_in) +
                          self.input_centrality_out(centrality_out))
        x = x + centrality_emb

        # 通过各层
        for i, layer in enumerate(self.layers):
            x = layer(x, spd_matrix, centrality_in, centrality_out, edge_encoding)
            if i < len(self.layers) - 1:  # 除最后一层外添加层间Dropout
                x = self.layer_dropout(x)

        return F.log_softmax(self.fc(x), dim=1)


# 7. 训练设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 计算最大中心性值
max_centrality = min(deg_in.max().item(), 50)  # 限制在50以内

model = Graphormer(
    input_dim=dataset.num_features,
    hidden_dim=64,
    output_dim=dataset.num_classes,
    num_layers=2,  # 优化为2层
    max_centrality=int(max_centrality),
    max_spd_value=10,  # 更合理的距离范围
    max_edge_encoding=5,  # 限制边编码范围
    drop_rate=0.5  # 更强正则化
).to(device)

# 转移数据到设备
spd_matrix = spd_matrix.to(device)
edge_encoding = edge_encoding.to(device)
deg_in = deg_in.to(device)
deg_out = deg_out.to(device)
data = data.to(device)

# 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,  # 更低学习率
    weight_decay=5e-4  # L2正则化
)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=20,

)


# 8. 训练循环
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, spd_matrix, deg_in, deg_out, edge_encoding)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return loss.item()


# 9. 测试函数
def test():
    model.eval()
    with torch.no_grad():
        logits = model(data.x, spd_matrix, deg_in, deg_out, edge_encoding)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].argmax(dim=1)
            correct = pred.eq(data.y[mask]).sum().item()
            total = mask.sum().item()
            acc = correct / total
            accs.append(acc)
    return accs


# 10. 可视化函数
def plot_training_curve(train_losses, val_accs, test_accs):
    plt.figure(figsize=(12, 4))

    # 训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 验证和测试准确率
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, 'g', label='Validation Accuracy')
    plt.plot(test_accs, 'r', label='Test Accuracy')
    plt.title('Validation and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('output/training_curve.png')
    plt.show()


# 11. 执行训练
print("Starting training...")
train_losses = []
val_accs = []
test_accs = []
best_test_acc = 0
start_time = time.time()

for epoch in range(300):
    loss = train()
    train_losses.append(loss)

    if epoch % 10 == 0:
        train_acc, val_acc, test_acc = test()
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        # 更新学习率
        scheduler.step(val_acc)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'output/best_graphormer_model.pth')

# 最终评估
train_acc, val_acc, test_acc = test()
print(f'\nTraining completed in {time.time() - start_time:.1f} seconds')
print(f'Best Test Accuracy: {best_test_acc:.4f}')
print(f'Final Performance:')
print(f'Train Accuracy: {train_acc:.4f}')
print(f'Validation Accuracy: {val_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

# 绘制训练曲线
plot_training_curve(train_losses, val_accs, test_accs)

# 加载最佳模型
model.load_state_dict(torch.load('output/best_graphormer_model.pth'))
print("Best model loaded for evaluation")


# 12. 模型分析和可视化
def analyze_model(model):
    print("\nModel Analysis:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 分析SPD嵌入权重
    spd_emb = model.layers[0].spd_embedding.weight.data.cpu()
    plt.figure(figsize=(10, 6))
    for head in range(model.layers[0].heads):
        plt.plot(spd_emb[:, head].numpy(), label=f'Head {head}')
    plt.title('SPD Embedding Weights by Distance')
    plt.xlabel('Shortest Path Distance')
    plt.ylabel('Embedding Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/spd_embeddings.png')
    plt.show()

    # 分析中心性嵌入权重
    cent_emb = model.input_centrality_in.weight.data.cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.plot(cent_emb)
    plt.title('Centrality Embeddings')
    plt.xlabel('Degree Centrality')
    plt.ylabel('Embedding Value')
    plt.grid(True)
    plt.savefig('output/centrality_embeddings.png')
    plt.show()

    # 分析边编码权重
    edge_emb = model.layers[0].edge_embedding.weight.data.cpu()
    plt.figure(figsize=(10, 6))
    for head in range(model.layers[0].heads):
        plt.plot(edge_emb[:, head].numpy(), label=f'Head {head}')
    plt.title('Edge Encoding Weights by Type')
    plt.xlabel('Edge Type (0=no,1=direct,>1=distance)')
    plt.ylabel('Embedding Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/edge_encodings.png')
    plt.show()


# 执行分析
analyze_model(model)