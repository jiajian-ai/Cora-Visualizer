import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches


def visualize_subgraph(data, num_nodes=50, save_path=None):
    # 随机选择num_nodes个节点
    subset = np.random.choice(data.num_nodes, num_nodes, replace=False)
    # 创建子图：我们需要边索引中两个节点都在子集内的那些边
    subset_mask = np.isin(data.edge_index[0].numpy(), subset) & np.isin(data.edge_index[1].numpy(), subset)
    edge_index_sub = data.edge_index[:, subset_mask]

    # 转换为networkx图
    G = to_networkx(data, to_undirected=False)
    # 创建子图
    G_sub = G.subgraph(subset)

    # 绘图
    plt.figure(figsize=(10, 7))
    pos = nx.random_layout(G_sub)

    # 节点颜色根据类别
    labels = data.y.numpy()
    colors = labels[list(G_sub.nodes)]

    nx.draw_networkx_nodes(G_sub, pos, node_size=50, node_color=colors, cmap=plt.cm.tab10)
    nx.draw_networkx_edges(G_sub, pos, width=0.5, alpha=0.5, arrowstyle='-|>', arrows=True, arrowsize=10)

    plt.title('Subgraph Visualization of Cora')
    plt.axis('off')

    # 添加图例
    label_names = [
        "Theory", "Reinforcement_Learning", "Genetic_Algorithms",
        "Neural_Networks", "Probabilistic_Methods", "Case_Based", "Rule_Learning"
    ]
    unique_labels = np.unique(labels[list(G_sub.nodes)])
    legend_patches = []
    for i, category_id in enumerate(unique_labels):
        color = plt.cm.tab10(category_id)
        legend_patches.append(mpatches.Patch(color=color, label=label_names[category_id]))
    plt.legend(handles=legend_patches, title="Node Categories", loc='upper left', ncol=2)

    # 添加子图统计信息
    plt.text(0.98, 0.95, f"Nodes: {G_sub.number_of_nodes()}\nEdges: {G_sub.number_of_edges()}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    plt.tight_layout()
    plt.show()


def visualize_tsne(data, save_path=None):
    # 提取节点特征和标签
    X = data.x.numpy()
    y = data.y.numpy()

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)

    # 可视化
    plt.figure(figsize=(12, 8))

    # 为每个类别创建颜色映射
    categories = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

    for i, category in enumerate(categories):
        plt.scatter(X_2d[y == category, 0], X_2d[y == category, 1],
                    color=colors[i], label=f'Class {category}', s=10, alpha=0.6)

    plt.title('Cora Dataset Node Visualization with t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # 添加图例（使用论文类别名称）
    label_names = [
        "Theory",
        "Reinforcement_Learning",
        "Genetic_Algorithms",
        "Neural_Networks",
        "Probabilistic_Methods",
        "Case_Based",
        "Rule_Learning"
    ]
    plt.legend(label_names)

    plt.grid(alpha=0.2)
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    import os
    
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    data = dataset[0]
    
    # 确保output文件夹存在
    os.makedirs('output', exist_ok=True)

    # 可视化子图并保存
    # visualize_subgraph(data, num_nodes=200, save_path='output/cora_subgraph.png')
    
    # # 可视化t-SNE并保存
    visualize_tsne(data, save_path='output/cora_tsne.png')