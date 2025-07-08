import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableShapelet(nn.Module):
    """
    一个模块，用于学习一组 Shapelet 并计算它们与输入时间序列的最小欧氏距离。
    该模块为多通道设计，并借鉴了您第二个 main.py 的核心思想。
    """

    def __init__(self, num_shapelets, in_channels, shapelet_length):
        """
        初始化可学习的 Shapelet 模块。

        参数:
            num_shapelets (int): 要学习的 Shapelet 的数量 (K)。
            in_channels (int): 输入时间序列的通道数 (M)。这是与参考代码的关键区别。
            shapelet_length (int): 每个 Shapelet 的长度 (L)。
        """
        super().__init__()
        self.num_shapelets = num_shapelets
        self.in_channels = in_channels
        self.shapelet_length = shapelet_length

        # 1. 将 Shapelet 初始化为可学习的 nn.Parameter
        # 形状为 (num_shapelets, in_channels, shapelet_length)，以支持多通道
        shapelets_tensor = torch.randn(num_shapelets, in_channels, shapelet_length)
        self.shapelets = nn.Parameter(shapelets_tensor, requires_grad=True)
        nn.init.uniform_(self.shapelets, -1, 1)  # 使用均匀分布进行初始化

    def _calculate_distances(self, x):
        """
        以高效、可微分的方式计算距离。

        参数:
            x (torch.Tensor): 输入时间序列，形状为 (batch_size, in_channels, seq_length)。

        返回:
            torch.Tensor: 每个样本与每个 Shapelet 的最小平方欧氏距离，形状为 (batch_size, num_shapelets)。
        """
        # 将输入 x 展开成所有可能的子序列
        # subsequences 形状: (batch_size, in_channels, num_subsequences, shapelet_length)
        subsequences = x.unfold(dimension=2, size=self.shapelet_length, step=1)

        # 调整维度以进行广播计算
        # sub_exp 形状: (batch_size, 1, in_channels, num_subsequences, shapelet_length)
        # shp_exp 形状: (1, num_shapelets, in_channels, 1, shapelet_length)
        sub_exp = subsequences.unsqueeze(1)
        shp_exp = self.shapelets.unsqueeze(0).unsqueeze(3)

        # 计算差的平方，然后对通道(dim=2)和长度(dim=4)维度求和
        # 得到每个子序列与每个 Shapelet 的平方欧氏距离
        # distances_sq 形状: (batch_size, num_shapelets, num_subsequences)
        distances_sq = torch.sum((sub_exp - shp_exp) ** 2, dim=(2, 4))

        # 找到每个 Shapelet 对应的最小距离
        # min_distances_sq 形状: (batch_size, num_shapelets)
        min_distances_sq, _ = torch.min(distances_sq, dim=2)

        return min_distances_sq

    def forward(self, x):
        """
        前向传播，计算最小距离特征。
        """
        return self._calculate_distances(x)

    # --- 正则化函数 ---
    # 注意：这些正则化函数与您第二个 main.py 中的 regConti 不同，
    # 因为我们没有实现其独特的 posMap 参数。这些是更通用的正则化项。

    def shape_regularization(self, x):
        """
        形态正则化：鼓励每个学习到的 Shapelet 接近训练集中的某个子序列。
        """
        min_dist_sq = self._calculate_distances(x)
        min_dist_per_shapelet, _ = torch.min(min_dist_sq, dim=0)
        reg_loss = torch.mean(min_dist_per_shapelet)
        return reg_loss

    def diversity_regularization(self):
        """
        多样性正则化：惩罚过于相似的 Shapelet 对。
        """
        flat_shapelets = self.shapelets.view(self.num_shapelets, -1)
        dist_matrix = torch.cdist(flat_shapelets, flat_shapelets, p=2)
        dist_matrix = dist_matrix + torch.eye(self.num_shapelets, device=dist_matrix.device) * 1e9
        diversity_loss = torch.exp(-dist_matrix).triu(diagonal=1).sum()
        num_pairs = self.num_shapelets * (self.num_shapelets - 1) / 2
        return diversity_loss / max(1, num_pairs)  # 避免除以零

# =============================================================================
# 这是根据流程图实现的新模型。
# =============================================================================

class LocalGlobalCrossAttentionModel(nn.Module):
    """
    实现流程图中的网络模型，通过交叉注意力（cross-attention）
    融合了局部的、基于shapelet的特征和全局的卷积特征。
    """

    def __init__(self, in_channels, num_shapelets, shapelet_length,
                 num_classes, embed_dim=64, n_heads=8):
        """
        参数:
            in_channels (int): 输入时间序列的通道数。
            seq_length (int): 输入时间序列的长度。
            num_shapelets (int): 要学习的shapelet数量（图中的 K）。
            shapelet_length (int): 每个shapelet的长度（图中的 l1）。
            num_classes (int): 用于分类的输出类别数。
            embed_dim (int): 注意力机制之前的特征嵌入维度。
            n_heads (int): 多头注意力的头数。
        """
        super().__init__()

        # --- Local 分支 (局部特征提取) ---
        self.shapelet_transformer = LearnableShapelet(
            num_shapelets=num_shapelets,
            in_channels=in_channels,
            shapelet_length=shapelet_length
        )
        self.local_encoder = nn.Sequential(
            nn.Linear(num_shapelets, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        # --- Global 分支 (全局特征提取) ---
        self.global_conv = nn.Sequential(
            # padding='same'，的意思是要求程序自动调整paddng大小，保证输入与输出的高度和宽度一致，所以(batch_size, 1, channels, seq_length) --> (batch_size, 16, channels, seq_length)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 5), padding='same'),
            nn.ReLU(),
            # stride默认下与卷积核大小一致，(batch_size, 16, channels, seq_length) --> (batch_size, 16, channels, seq_length/2)
            nn.MaxPool2d(kernel_size=(1, 2)),
            # padding='same'，所以(batch_size, 16, channels, seq_length/2) --> (batch_size, 32, channels, seq_length/2)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), padding='same'),
            nn.ReLU(),
            #(batch_size, 32, channels, seq_length/2) --> (batch_size, 32, channels, seq_length/4)
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        self.global_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        # --- 特征融合 (交叉注意力) ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )

        # --- 最终分类器 (MLP) ---
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        模型的正向传播。
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, seq_length)。
        返回:
            torch.Tensor: 分类的logits（原始分数），形状为 (batch_size, num_classes)。
        """
        # --- Local 分支路径 ---
        shapelet_features = self.shapelet_transformer(x)  # (批次, shapelet数量)
        local_features = self.local_encoder(shapelet_features)  # (批次, embed_dim)

        # --- Global 分支路径 ---
        # 增加一个维度以适应 Conv2D
        x_2d = x.unsqueeze(1)  # (批次, 1, 通道数, 序列长度)
        conv_maps = self.global_conv(x_2d)  # (批次, 32, 通道数, 序列长度/4)
        global_features = self.global_encoder(conv_maps)  # (批次, embed_dim)

        # --- 通过交叉注意力进行融合 ---
        # Q = [local features], K=V = [global features]
        # 为 batch_first=True 的注意力机制调整形状: (批次, 序列长度, 特征数)
        query = local_features.unsqueeze(1)  # (批次, 1, embed_dim)
        key = global_features.unsqueeze(1)  # (批次, 1, embed_dim)
        value = global_features.unsqueeze(1)  # (批次, 1, embed_dim)

        attn_output, _ = self.cross_attention(query, key, value)
        fused_features = attn_output.squeeze(1)  # (批次, embed_dim)

        # --- 最终分类 ---
        logits = self.mlp(fused_features)  # (批次, 类别数)

        return logits


if __name__ == '__main__':
    # --- 使用示例 ---
    BATCH_SIZE = 32
    IN_CHANNELS = 3  # 例如，一个三轴加速度计信号
    SEQ_LENGTH = 256
    NUM_CLASSES = 6

    NUM_SHAPELETS = 50
    SHAPELET_LENGTH = 30
    EMBED_DIM = 128
    N_HEADS = 8

    # 创建一个虚拟输入张量
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LENGTH)

    # 实例化模型
    model = LocalGlobalCrossAttentionModel(
        in_channels=IN_CHANNELS,
        seq_length=SEQ_LENGTH,
        num_shapelets=NUM_SHAPELETS,
        shapelet_length=SHAPELET_LENGTH,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS
    )

    # 执行一次前向传播
    output = model(dummy_input)

    # 打印模型结构和输出形状
    print("--- 模型测试 ---")
    print(f"输入形状:  {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    # 检查输出形状是否正确
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print("\n✅ 测试通过: 输出形状正确。")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总可训练参数量: {total_params:,}")