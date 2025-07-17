# --- START OF COMBINED AND SIMPLIFIED FILE ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# 实用工具类 (Utility Class)
# ----------------------------------------------------------------------------
@dataclass
class ModelInfo:
    """
    一个简单的数据类，用于存储和传递模型在正向传播过程中的各种中间结果和最终输出。
    """
    d: torch.Tensor  # shapelet与时间序列子序列之间的最小距离
    p: torch.Tensor  # shapelet变换后的概率/谓词
    shapelet_preds: torch.Tensor  # Shapelet分支的直接预测结果
    preds: torch.Tensor  # 模型的最终预测输出
    loss: torch.Tensor  # 模型计算出的正则化和多样性损失


# ----------------------------------------------------------------------------
# Shapelet 核心模块 (Shapelet Core Module)
# ----------------------------------------------------------------------------

class Shapelet(nn.Module):
    """
    简化的Shapelet类。
    - 距离函数固定为L1-like的平均绝对差值。
    - 不再需要'distance_func'和'memory_efficient'等参数。
    """

    def __init__(self, dim_data, shapelet_len, seq_length, num_shapelet=10, stride=1, eps=1.):
        super().__init__()

        self.dim = dim_data
        self.length = shapelet_len
        self.n = num_shapelet
        self.stride = stride
        torch.manual_seed(42)
        # shapelet的shape：(num_shapelet, c, l)
        self.weights = nn.Parameter(torch.normal(0, 1, (self.n, self.dim, self.length)), requires_grad=True)
        self.eps = eps
        # --- 【新增】可学习的惩罚/重要性矩阵 ---
        # 计算滑动窗口产生的子序列数量 m
        num_subsequences = (seq_length - self.length) // self.stride + 1
        # 创建一个可学习的参数，形状为(通道数, 子序列数)
        # 初始化为1，表示初始状态下没有惩罚
        self.position_channel_map = nn.Parameter(torch.ones(self.dim, num_subsequences), requires_grad=True)
    def forward(self, x):
        """
        unfold(对哪一个维度进行操作，滑动窗口长度，步长)。ShapeBottleneckModel已经x = rearrange(x, 'b t c -> b c t')，所以是对第二个维度做滑动窗口
        x变化：(b,c,t) --> (b,c,m,self.length)  其中m=floor((t-self.length)/self.stride)+1  通过滑动窗口切割后子序列的数量
        """
        x = x.unfold(2, self.length, self.stride)

        # shape变化：(b,c,m,l) --> (b,m,1,c,l)
        x = rearrange(x, 'b c m l -> b m 1 c l')  # .permute((0, 2, 1, 3)).unsqueeze(2)#.contiguous()

        # 距离计算被简化，现在只使用平均绝对差值。
        """
        广播机制：
        广播前：
        x: (b,m,1,c,l)      self.weights: (n,c,l)
        广播后：
        x: (b,m,n,c,l)      self.weights: (b,m,n,c,l)

        mean(dim=-1) 会消除最后一个维度
        d：(b,m,n,c)
        """
        d = (x - self.weights).abs().mean(dim=-1)
        # --- 【修改】应用惩罚矩阵 ---
        # self.position_channel_map 的形状是 (c, m)
        # d 的形状是 (b, m, n, c)
        # 我们需要将 map 的形状调整为可以与 d 进行广播相乘
        # (c, m) -> permute(1,0) -> (m, c) -> unsqueeze -> (1, m, 1, c)
        # 这样它就可以和 (b, m, n, c) 的 d 进行广播相乘了
        penalty_map = self.position_channel_map.permute(1, 0).unsqueeze(0).unsqueeze(2)

        # 将惩罚值与距离相乘。惩罚值越小，代表此位置越重要，距离也越小
        d = d * penalty_map


        # Maximum rbf prob，高斯核  p: (b,m,n,c)
        p = torch.exp(-torch.pow(self.eps * d, 2))  # RBF

        # 直通估计 (Straight-through estimator)
        #这三句代码的作用放在了Straight-Through Estimator(STE).txt文件中进行说明

        """hard: (b, m, n, c)"""
        hard = torch.zeros_like(p).scatter_(1, p.argmax(dim=1, keepdim=True), 1.)
        """soft: (b, m, n, c)"""
        soft = torch.softmax(p, dim=1)
        """onehot_max：(b, m, n, c)"""
        onehot_max = hard + soft - soft.detach()



        """max_p:(b,n,c) """
        max_p = torch.sum(onehot_max * p, dim=1)

        """展平操作，max_p:(b,n,c) --> max_p.flatten(start_dim=1): (b,n*c) """
        return max_p.flatten(start_dim=1), d.min(dim=1).values.flatten(start_dim=1)

    # 【新增】step方法用于裁剪惩罚矩阵
    def step(self):
        # 裁剪惩罚矩阵，使其值非负，这保证了惩罚的直观意义
        with torch.no_grad():
            self.position_channel_map.clamp_(min=0.)
    def derivative(self):
        # 计算shapelet的导数（相邻点之差），可用于可视化或分析
        return torch.diff(self.weights, dim=-1)


# ----------------------------------------------------------------------------
# Shapelet 瓶颈模型 (Shapelet Bottleneck Model)
# ----------------------------------------------------------------------------

class ShapeBottleneckModel(nn.Module):
    """
    简化的ShapeBottleneckModel。
    - 分类器固定为'linear'层, 不再需要根据configs.sbm_cls进行分支判断。
    - DistThresholdSBM 和 SelfAttention 等相关模块已被移除。
    """

    def __init__(self,in_channels,seq_length,num_classes,num_shapelet=[5, 5, 5, 5],shapelet_len=[0.1, 0.2, 0.3, 0.5]):
        super().__init__()

        self.num_channel = in_channels
        self.num_class = num_classes
        self.shapelet_len = []
        self.seq_length = seq_length

        # 初始化 shapelets
        self.shapelets = nn.ModuleList()
        for i, l in enumerate(shapelet_len):
            # 根据配置的比例计算每个shapelet的实际长度sl
            sl = max(3, np.ceil(l * seq_length).astype(int))
            # Shapelet实例化过程被简化
            self.shapelets.append(
                Shapelet(
                    dim_data=self.num_channel,
                    shapelet_len=sl,
                    seq_length=self.seq_length,
                    num_shapelet=num_shapelet[i],
                    eps=1.,
                    stride=1 if seq_length < 3000 else max(1, int(np.log2(sl)))
                )
            )
            self.shapelet_len.append(sl)

        self.total_shapelets = sum(num_shapelet * self.num_channel)

        # 初始化分类器 - 简化为只使用一个线性层
        self.output_layer = nn.Linear(self.total_shapelets, self.num_class, bias=False)

        self.dropout = nn.Dropout(0.5)
        self.distance_func = nn.PairwiseDistance(p=2)  # 用于多样性损失的距离度量
        self.lambda_reg = 0.1  # 分类器权重的L1正则化系数
        self.lambda_div = 0.1  # Shapelet多样性损失系数

    def forward(self, x, *args, **kwargs):
        # 维度重排以匹配shapelet处理的格式
        #x = rearrange(x, 'b t c -> b c t')
        # 实例归一化 (Instance normalization)
        x = (x - x.mean(dim=-1, keepdims=True)) / (x.std(dim=-1, keepdims=True) + 1e-8)

        # 通过Shapelet变换获得谓词
        shapelet_probs, shapelet_dists = [], []

        """p、d ： (b,n*c)  -->  shapelet_probs、shapelet_dists(b,n*c*len(shapelet_len))"""
        for shapelet in self.shapelets:
            p, d = shapelet(x)
            shapelet_probs.append(p)
            shapelet_dists.append(d)
        shapelet_probs = torch.cat(shapelet_probs, dim=-1)
        shapelet_dists = torch.cat(shapelet_dists, dim=-1)

        # 预测 - 逻辑被简化，直接通过线性层输出
        out = self.output_layer(self.dropout(shapelet_probs))

        return out, ModelInfo(d=shapelet_dists,
                              p=shapelet_probs,
                              shapelet_preds=out,
                              preds=out,
                              loss=self.loss().unsqueeze(0))

    def step(self):
        # 在每步优化后，将分类器权重裁剪为非负值，有助于模型的可解释性
        with torch.no_grad():
            self.output_layer.weight.clamp_(0.)

    def loss(self):
        # 计算模型的总损失
        # L1正则化损失
        loss_reg = self.output_layer.weight.abs().mean()
        # Shapelet多样性损失
        loss_div = self.diversity() if self.lambda_div > 0. else 0.
        return loss_reg * self.lambda_reg + loss_div * self.lambda_div

    def diversity(self):
        # 计算shapelet之间的多样性，鼓励学习到不同模式的shapelet
        loss = 0.
        for s in self.shapelets:
            sh = s.weights.permute(1, 0, 2)
            dist = self.distance_func(sh.unsqueeze(1), sh.unsqueeze(2))
            mask = torch.ones_like(dist) - torch.eye(sh.shape[1], device=dist.device).unsqueeze(0)
            loss += (torch.exp(-dist) * mask).mean()
        return loss

    def get_shapelets(self):
        # 从模型中提取所有学习到的shapelet
        shapelets = []
        for s in self.shapelets:
            for k in range(s.weights.data.shape[0]):
                for c in range(s.weights.data.shape[1]):
                    """组成了一个元组，(l,c) l代表shapelet的长度，c代表这个shapelet所在的通道"""
                    # 返回一个列表，其中每个元素是一个元组 (shapelet数据, 通道索引)
                    shapelets.append((s.weights.data[k, c, :].cpu().numpy(), c))
        return shapelets


# ----------------------------------------------------------------------------
# 主模型 InterpGN (Main Model)
# ----------------------------------------------------------------------------

class InterpGN(nn.Module):
    """
    简化的InterpGN模型。
    此版本被精简，仅使用ShapeBottleneckModel (SBM) 分支。
    并行的深度学习模型 (`deep_model`) 和Gini指数门控机制已被移除。
    """

    def __init__(
            self,
            in_channels,
            seq_length,
            num_classes,
            num_shapelet=[5, 5, 5, 5],
            shapelet_len=[0.1, 0.2, 0.3, 0.5],

    ):
        super().__init__()
        self.sbm = ShapeBottleneckModel(
            num_shapelet= num_shapelet,
            shapelet_len= shapelet_len,
            num_classes=num_classes,
            in_channels = in_channels,
            seq_length = seq_length
        )
        # deep_model 已被移除

    def forward(self, x):
        """
        前向传播被简化，仅通过SBM处理数据。
        - 之前用于deep_model的参数（如x_mark_enc）已被移除。
        - Gini指数计算和门控逻辑也已被移除。
        - 直接返回SBM的输出和信息。
        """
        sbm_out, model_info = self.sbm(x)
        return sbm_out, model_info

    def loss(self):
        # 损失计算完全来自SBM组件
        return self.sbm.loss()

    def step(self):
        # step操作代理给SBM组件
        self.sbm.step()

# --- END OF COMBINED AND SIMPLIFIED FILE ---