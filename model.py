import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 这部分代码直接取自您提供的 model.py 文件。
# 它实现了图中“local”分支的核心——形态基元（shapelet）发现机制。
# 对应图中的: 可学习张量 -> shapelets -> (部分) local features
# =============================================================================

class ShapeConv(nn.Module):
    def __init__(self, in_channels, out_channels, shapelet_length, num_classes=None, supervised=True):
        """
        初始化ShapeConv层。

        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数（即shapelet的数量）。
            shapelet_length (int): 每个shapelet的长度。
            num_classes (int): 分类任务的类别数（在监督学习时使用）。
            supervised (bool): 是否为监督学习模式的标志。
        """
        super(ShapeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapelet_length = shapelet_length
        self.num_classes = num_classes
        self.supervised = supervised

        # 定义卷积核（即shapelets）
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=shapelet_length,
            stride=1,
            padding=0,
            bias=False
        )
        self._initialize_shapelets()

    def _initialize_shapelets(self):
        """初始化shapelet权重（监督或无监督模式）。"""
        with torch.no_grad():
            if self.supervised and self.num_classes is not None:
                k = self.out_channels // self.num_classes
                for i in range(self.num_classes):
                    for j in range(k):
                        torch.manual_seed(42 + i * k + j)
                        idx = i * k + j
                        weight_slice = self.conv.weight.data[idx]
                        nn.init.uniform_(weight_slice, a=-2.0, b=2.0)
            else:
                self.conv.weight.data.normal_(0, 0.1)

    def shape_regularization(self, x):
        """以向量化且可微分的方式计算shape正则化损失。"""
        conv_out = self.conv(x)
        shapelet_norm_sq = torch.sum(self.conv.weight ** 2, dim=(1, 2))
        sub_sequences = x.unfold(dimension=2, size=self.shapelet_length, step=1)
        sub_sequences_norm_sq = torch.sum(sub_sequences ** 2, dim=(1, 3))
        norm_term = shapelet_norm_sq.view(1, -1, 1) + sub_sequences_norm_sq.unsqueeze(1)
        shapelet_distances = norm_term - 2 * conv_out
        min_distances, _ = torch.min(shapelet_distances, dim=2)
        reg_loss = torch.mean(min_distances)
        return reg_loss

    def diversity_regularization(self):
        """计算多样性正则化项，防止shapelet之间过于相似。"""
        weights = self.conv.weight.view(self.out_channels, -1)
        # 计算所有 shapelet 两两之间的欧氏距离矩阵
        dist_matrix = torch.cdist(weights, weights, p=2)
        # 使用指数函数惩罚距离近的 shapelet 对，并取均值
        diversity_loss = torch.exp(-dist_matrix).mean()
        return diversity_loss

    def forward(self, x):
        """
        前向传播，计算shapelet变换。
        这里计算的是一个与最小平方欧氏距离成正比的值。
        """
        conv_out = self.conv(x)
        shapelet_norm = torch.sum(self.conv.weight ** 2, dim=(1, 2)) / 2
        batch_size, _, seq_length = x.shape
        sub_sequences = x.unfold(dimension=2, size=self.shapelet_length, step=1)
        input_norm = torch.sum(sub_sequences ** 2, dim=(1, 3)) / 2

        shapelet_norm = shapelet_norm.view(1, -1, 1)
        input_norm = input_norm.view(batch_size, 1, -1)
        norm_term = shapelet_norm + input_norm

        features = conv_out - norm_term
        features = F.max_pool1d(features, kernel_size=features.size(-1)).squeeze(-1)
        features = -2 * features
        return features


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
        self.shapelet_transformer = ShapeConv(
            in_channels=in_channels, out_channels=num_shapelets,
            shapelet_length=shapelet_length, num_classes=num_classes
        )
        self.local_encoder = nn.Sequential(
            nn.Linear(num_shapelets, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        # --- Global 分支 (全局特征提取) ---
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        # --- 错误修复点 ---
        # 最后一个Conv2d的输出通道是32，经过池化和展平后，送入线性层的特征数就是32。
        # 原来的 `32 * in_channels` 是错误的。
        self.global_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embed_dim),  # 将输入维度从 32 * in_channels 修改为 32
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