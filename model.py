import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableShapelet1D(nn.Module):
    """
    学习一组一维 Shapelet，并通过聚合各通道距离的方式与多通道时间序列进行匹配。
    """

    def __init__(self, num_shapelets, shapelet_length, in_channels, distance_aggregation='sum'):
        """
        参数:
            num_shapelets (int): Shapelet 的数量。
            shapelet_length (int): Shapelet 的长度。
            in_channels (int): 输入数据的通道数，用于计算距离。
            distance_aggregation (str): 通道距离的聚合方式, 'sum' 或 'min'。
        """
        super().__init__()
        self.num_shapelets = num_shapelets
        self.shapelet_length = shapelet_length
        self.in_channels = in_channels
        self.distance_aggregation = distance_aggregation

        # 核心修改：Shapelet 现在是一维的
        # 形状为 (num_shapelets, shapelet_length)
        shapelets_tensor = torch.randn(num_shapelets, shapelet_length)
        self.shapelets = nn.Parameter(shapelets_tensor, requires_grad=True)
        nn.init.uniform_(self.shapelets, -1, 1)

    def _calculate_distances(self, x):
        """
        高效地计算一维 Shapelet 与多通道序列的距离。
        """
        # x 形状: (B, M, Q) -> B=batch, M=in_channels, Q=seq_length
        # subsequences 形状: (B, M, N, L) -> N=num_subsequences, L=shapelet_length
        subsequences = x.unfold(dimension=2, size=self.shapelet_length, step=1)

        # self.shapelets 形状: (K, L) -> K=num_shapelets

        # 扩展维度以进行广播计算
        # sub_exp 形状: (B, M, N, 1, L)
        # shp_exp 形状: (1, 1, 1, K, L)
        sub_exp = subsequences.unsqueeze(3)
        shp_exp = self.shapelets.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # 计算差的平方，然后对长度维度(dim=4)求和
        # 得到每个通道上，每个子序列与每个 Shapelet 的距离
        # dist_per_channel 形状: (B, M, N, K)
        dist_per_channel = torch.sum((sub_exp - shp_exp) ** 2, dim=4)

        # 核心修改：聚合来自不同通道(M, dim=1)的距离
        if self.distance_aggregation == 'sum':
            # 将各通道的距离相加
            agg_dist = torch.sum(dist_per_channel, dim=1)
        elif self.distance_aggregation == 'min':
            # 找出在哪个通道上匹配得最好（距离最小）
            agg_dist, _ = torch.min(dist_per_channel, dim=1)
        else:
            raise ValueError("distance_aggregation must be 'sum' or 'min'")

        # agg_dist 形状: (B, N, K)

        # 在所有子序列(N, dim=1)中找到最小距离
        min_dist_sq, _ = torch.min(agg_dist, dim=1)
        # min_dist_sq 形状: (B, K)

        return min_dist_sq

    def forward(self, x):
        return self._calculate_distances(x)

    def diversity_regularization(self):
        """多样性正则化，与之前类似，但现在作用于一维 Shapelet。"""
        # self.shapelets 形状: (K, L)
        dist_matrix = torch.cdist(self.shapelets, self.shapelets, p=2)
        dist_matrix = dist_matrix + torch.eye(self.num_shapelets, device=dist_matrix.device) * 1e9
        diversity_loss = torch.exp(-dist_matrix).triu(diagonal=1).sum()
        num_pairs = self.num_shapelets * (self.num_shapelets - 1) / 2
        return diversity_loss / max(1, num_pairs)

    def shape_regularization(self, x):
        """形态正则化也需要遵循新的距离计算逻辑。"""
        min_dist_sq = self._calculate_distances(x)  # (B, K)
        min_dist_per_shapelet, _ = torch.min(min_dist_sq, dim=0)  # (K)
        reg_loss = torch.mean(min_dist_per_shapelet)
        return reg_loss

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
        self.shapelet_transformer = LearnableShapelet1D(
            num_shapelets=num_shapelets,
            shapelet_length=shapelet_length,
            in_channels=in_channels,
            distance_aggregation='min'  # 您可以在这里选择 'sum' 或 'min'
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