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
        initial_tensor = torch.empty(shapelet_length)
        # 2. 使用你想要的分布（uniform）来初始化它
        nn.init.uniform_(initial_tensor, -1, 1)
        # 3. 封装成 nn.Parameter
        self.shapelet = nn.Parameter(initial_tensor, requires_grad=True)

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
# Main Model: LocalShapeletModel
# The final model framework, containing only the local branch.
# =============================================================================

class LocalShapeletModel(nn.Module):
    """
    A model that uses only the local branch (learnable shapelets with penalty maps)
    for time series classification.
    """

    def __init__(self, in_channels, seq_length, num_shapelets, shapelet_length, num_classes):
        super().__init__()

        SEED = 42
        torch.manual_seed(SEED)

        # --- Local 分支 (The only branch) ---
        self.shapelet_transformer = LearnableShapeletWithPenalty(
            num_shapelets=num_shapelets,
            shapelet_length=shapelet_length,
            in_channels=in_channels,
            seq_length=seq_length
        )

        # --- 最终分类器 (MLP) ---
        # The input to the MLP is now the direct output of the shapelet transformer,
        # which has a dimension of `num_shapelets`.
        hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(num_shapelets, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        The model's forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length).
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes).
        """
        # --- Local 分支路径 ---
        # Get shapelet features of shape (batch_size, num_shapelets)
        shapelet_features = self.shapelet_transformer(x)

        # --- 最终分类 ---
        # Feed features directly into the MLP
        logits = self.mlp(shapelet_features)

        return logits