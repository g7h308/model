import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Local Branch Core Component: ShapeletBlock
# 管理单个一维Shapelet及其对应的二维(通道-位置)惩罚/权重矩阵。
# =============================================================================

class ShapeletBlock(nn.Module):
    """
    管理单个一维 Shapelet 及其对应的二维(通道-位置)惩罚矩阵。
    """

    def __init__(self, shapelet_length, in_channels, seq_length):
        super().__init__()
        self.shapelet_length = shapelet_length
        self.in_channels = in_channels

        # 1. 定义一维 Shapelet
        self.shapelet = nn.Parameter(torch.randn(shapelet_length), requires_grad=True)
        nn.init.uniform_(self.shapelet, -1, 1)

        # 2. 定义二维惩罚/权重图
        num_positions = seq_length - shapelet_length + 1  #shapelet的起始位置只能从0到seq_length - shapelet_length + 1上进行选择
        # penalty_map : (in_channels, num_positions)   #初始惩罚值全为0，标明起始对所有通道、所有位置都是平等的
        self.penalty_map = nn.Parameter(torch.zeros(in_channels, num_positions), requires_grad=True)

    def forward(self, subsequences):
        """
        计算加权距离。

        参数:
            subsequences (torch.Tensor): 形状为 (B, M, N, L) 的输入子序列。
                                         B=batch, M=in_channels, N=num_positions, L=shapelet_length

        返回:
            torch.Tensor: 对所有子序列计算的加权距离，形状为 (B, N)。
        """
        # self.shapelet 形状: (L) -> 扩展后 (1, 1, 1, L)
        shp_exp = self.shapelet.view(1, 1, 1, -1)

        # 计算每个通道、每个位置的原始平方距离
        # raw_dist_sq 形状: (B, M, N)
        raw_dist_sq = torch.sum((subsequences - shp_exp) ** 2, dim=3)

        # 计算惩罚/权重因子, 使用 elu(-m)+1 形式确保权重为正且初始接近1
        # penalty 形状: (M, N) -> 扩展后 (1, M, N)
        penalty = F.elu(-self.penalty_map) + 2.0
        penalty_exp = penalty.unsqueeze(0)

        # 将原始距离与惩罚/权重相乘
        # weighted_dist_sq 形状: (B, M, N)
        weighted_dist_sq = raw_dist_sq * penalty_exp

        # 将所有通道的加权距离相加，得到每个位置的总加权距离
        # total_weighted_dist 形状: (B, N)
        total_weighted_dist = torch.sum(weighted_dist_sq, dim=1)

        return total_weighted_dist


# =============================================================================
# Local Branch Main Module: LearnableShapeletWithPenalty
# 作为所有ShapeletBlock的容器，并整合所有正则化方法。
# =============================================================================

class LearnableShapeletWithPenalty(nn.Module):
    def __init__(self, num_shapelets, shapelet_length, in_channels, seq_length):
        super().__init__()
        self.num_shapelets = num_shapelets
        self.shapelet_length = shapelet_length

        # 创建一个 ModuleList 来容纳所有的 ShapeletBlock
        self.shapelet_blocks = nn.ModuleList(
            [ShapeletBlock(shapelet_length, in_channels, seq_length) for _ in range(num_shapelets)]
        )

    def forward(self, x):
        """
        前向传播，计算所有 Shapelet 的最小加权距离特征。
        """
        # 预先展开输入 x，所有 block 共享, x将会转化为不同开始位置的子序列

        # subsequences 形状: (B, M, N, L)   其中M=in_channels, N = seq_length - shapelet_length+1
        subsequences = x.unfold(dimension=2, size=self.shapelet_length, step=1)

        all_min_distances = []
        for block in self.shapelet_blocks:
            # total_weighted_dist 形状: (B, N)
            total_weighted_dist = block(subsequences)

            # 找到当前 Shapelet 在所有位置上的最小加权距离
            min_dist, _ = torch.min(total_weighted_dist, dim=1)
            all_min_distances.append(min_dist)

        # 将所有最小距离堆叠成最终的特征向量
        # shapelet_features 形状: (B, K)
        shapelet_features = torch.stack(all_min_distances, dim=1)

        return shapelet_features

    def diversity_regularization(self):
        """对所有一维 Shapelet 计算多样性"""
        if self.num_shapelets <= 1:
            return 0.0

        all_shapelets = torch.stack([block.shapelet for block in self.shapelet_blocks])
        dist_matrix = torch.cdist(all_shapelets, all_shapelets, p=2)
        dist_matrix = dist_matrix + torch.eye(self.num_shapelets, device=dist_matrix.device) * 1e9
        diversity_loss = torch.exp(-dist_matrix).triu(diagonal=1).sum()
        num_pairs = self.num_shapelets * (self.num_shapelets - 1) / 2
        return diversity_loss / num_pairs

    def shape_regularization(self, x):
        """形态正则化，基于无惩罚的原始距离"""
        subsequences = x.unfold(dimension=2, size=self.shapelet_length, step=1)
        total_reg_loss = 0.0

        for block in self.shapelet_blocks:
            shp_exp = block.shapelet.view(1, 1, 1, -1)
            # 聚合通道和长度维度，得到原始平方距离
            raw_dist_sq_per_pos = torch.sum((subsequences - shp_exp) ** 2, dim=(1, 3))  # (B, N)
            min_raw_dist, _ = torch.min(raw_dist_sq_per_pos, dim=1)  # (B,)
            total_reg_loss += torch.mean(min_raw_dist)

        return total_reg_loss / self.num_shapelets

    def penalty_regularization(self, lambda_l1=1e-5, lambda_l2=1e-5):
        """对 penalty_map 施加平滑性正则化"""
        reg_loss = 0.0
        for block in self.shapelet_blocks:
            pmap = block.penalty_map  # (M, N)
            # 位置平滑性 (沿 N 维度)
            if pmap.shape[1] > 1:
                pos_diff = pmap[:, 1:] - pmap[:, :-1]
                reg_loss += lambda_l2 * torch.sum(pos_diff ** 2) + lambda_l1 * torch.sum(torch.abs(pos_diff))
            # 通道平滑性 (沿 M 维度)
            if pmap.shape[0] > 1:
                chan_diff = pmap[1:, :] - pmap[:-1, :]
                reg_loss += lambda_l2 * torch.sum(chan_diff ** 2) + lambda_l1 * torch.sum(torch.abs(chan_diff))

        return reg_loss / self.num_shapelets


# =============================================================================
# Main Model: LocalGlobalCrossAttentionModel
# 最终的模型框架，整合了局部和全局分支。
# =============================================================================

class LocalGlobalCrossAttentionModel(nn.Module):
    """
    通过交叉注意力融合局部（基于加权Shapelet）和全局（基于CNN）特征的模型。
    """

    def __init__(self, in_channels, seq_length, num_shapelets, shapelet_length,
                 num_classes, embed_dim=64, n_heads=8):
        super().__init__()

        # --- Local 分支 (使用带有惩罚图的最终模块) ---
        self.shapelet_transformer = LearnableShapeletWithPenalty(
            num_shapelets=num_shapelets,
            shapelet_length=shapelet_length,
            in_channels=in_channels,
            seq_length=seq_length  # 新模块需要 seq_length 来确定惩罚图大小
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
        shapelet_features = self.shapelet_transformer(x)
        local_features = self.local_encoder(shapelet_features)

        # --- Global 分支路径 ---
        x_2d = x.unsqueeze(1)
        conv_maps = self.global_conv(x_2d)
        global_features = self.global_encoder(conv_maps)

        # --- 通过交叉注意力进行融合 ---
        query = local_features.unsqueeze(1)
        key = global_features.unsqueeze(1)
        value = global_features.unsqueeze(1)
        attn_output, _ = self.cross_attention(query, key, value)
        fused_features = attn_output.squeeze(1)

        # --- 最终分类 ---
        logits = self.mlp(fused_features)

        return logits