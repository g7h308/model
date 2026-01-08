import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# --- 1. 导入项目模块 ---
from InterpGN import InterpGN

try:
    # 尝试从 dataloader.py 导入 load 函数
    from dataloader import load as load_dataset
except ImportError:
    print("Error: Could not import 'load' from dataloader.py.")
    print("Please ensure dataloader.py is in the same directory.")
    exit()


def smooth_array(x, w_size=5):
    """
    对序列 x 进行滑动窗口平滑处理 (Moving Average)
    原始代码逻辑复现
    """
    if w_size >= len(x):
        return x

    out = np.zeros_like(x)
    for i in range(len(x)):
        # 确定窗口范围，处理边界情况
        start = max(0, i - w_size // 2)
        end = min(len(x), i + w_size // 2 + 1)
        out[i] = np.mean(x[start:end])
    return out

# ==============================================================================
# 2. 核心可视化函数 (针对 InterpGN 适配)
# ==============================================================================

def get_effective_weights(model):
    """
    计算 MLP 分类头的等效线性权重。
    InterpGN 的 output_layer 是 Sequential(Linear -> ReLU -> Dropout -> Linear)。
    我们需要合并两个 Linear 层的权重来得到 (num_classes, total_shapelets) 的映射。
    """
    # 获取 SBM 中的 output_layer (Sequential)
    # 结构: [0]Linear -> [1]LeakyReLU -> [2]Dropout -> [3]Linear
    # 根据 InterpGN.py
    layer1 = model.sbm.output_layer[0]
    layer2 = model.sbm.output_layer[3]

    w1 = layer1.weight.detach()  # Shape: (hidden, total_shapelets)
    w2 = layer2.weight.detach()  # Shape: (num_classes, hidden)

    # 矩阵乘法合并: (num_classes, hidden) @ (hidden, total) -> (num_classes, total)
    effective_weights = torch.matmul(w2, w1)
    return effective_weights


def visualize_shapelets_adapted(model, x_data, y_label, sample_id=0, top_k=5, save_path=None, do_smooth = True):
    """
    针对 InterpGN 修改版的可视化函数。

    参数:
    - model: 训练好的 InterpGN 模型
    - x_data: 测试数据 Tensor (B, C, T)
    - y_label: 真实标签 Tensor (B,)
    - sample_id: 要画 batch 中的第几个样本
    - top_k: 画几个最重要的 Shapelet
    - save_path: 图片保存路径 (可选)
    """
    model.eval()
    device = next(model.parameters()).device

    # --- 数据准备 ---
    x_sample = x_data[sample_id]  # (C, T)
    label = y_label[sample_id].item()

    # 构造 batch 输入
    x_tensor = x_sample.unsqueeze(0).to(device)  # (1, C, T)

    print(f"[-] Processing Sample ID: {sample_id}, True Class: {label}, Input Shape: {x_tensor.shape}")

    # --- 前向传播 ---
    with torch.no_grad():
        # InterpGN forward 返回 (output, model_info)
        _, model_info = model(x_tensor)
        # p shape: (1, total_shapelets)
        # 这里的 p 已经经过 RBF 和 sum，隐含了位置信息
        p = model_info.p.cpu()

    # --- 计算局部重要性 (Local Explanation) ---
    all_rules = get_effective_weights(model).cpu()
    rule = all_rules[label, :]

    # 贡献分 = 权重 * 激活值
    logits = rule * p.squeeze()

    # 排序
    top_indices = torch.argsort(logits, descending=True)[:top_k]
    print(f"    Top-{top_k} Shapelet Indices: {top_indices.numpy()}")

    # --- 提取 Shapelet 信息 ---
    shapelets_list = []

    # 遍历顺序必须与 forward 中的 concat 顺序一致
    for s_module in model.sbm.shapelets:
        weights = s_module.weights.detach().cpu()  # (num, dim, len)
        p_map = s_module.position_channel_map.detach().cpu()  # (dim, num_subsequences)
        stride = s_module.stride

        for k in range(weights.shape[0]):  # dim: num_shapelet
            for c in range(weights.shape[1]):  # dim: channel
                s_data = weights[k, c, :].numpy()
                s_penalty = p_map[c, :].numpy()
                shapelets_list.append({
                    'data': s_data,
                    'channel': c,
                    'stride': stride,
                    'penalty': s_penalty
                })

    # --- 绘图 ---
    x_np = x_sample.cpu().numpy()  # (C, T)
    num_channels = x_np.shape[0]

    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 1.0 * num_channels), sharex=True)
    if num_channels == 1: axs = [axs]

    # 1. 绘制背景（原始序列）
    for c in range(num_channels):
        axs[c].plot(x_np[c], color='gray', alpha=0.3, linewidth=1)
        axs[c].set_ylabel(f'Ch{c}', rotation=0, labelpad=10, fontsize=8)
        axs[c].set_yticks([])
        if c == 0:
            axs[c].set_title(f"Sample {sample_id} (Class {label})", fontsize=10)

    # 2. 绘制高亮 Shapelet
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for rank, idx in enumerate(top_indices):
        idx = idx.item()
        if idx >= len(shapelets_list): continue

        s_info = shapelets_list[idx]
        s_data = s_info['data']
        s_channel = s_info['channel']
        s_stride = s_info['stride']
        s_penalty = s_info['penalty']  # (num_subsequences,)
        s_len = len(s_data)

        if do_smooth:
            # 窗口大小设为 5 (或者 int(len(s_data) * 0.1))，可根据需求调整
            s_data = smooth_array(s_data, w_size=int(len(s_data) * 0.1))

        # 注意：建议用“平滑后”的 s_data 或者是“原始” s_data 来计算距离？
        # 原始逻辑通常是：用原始参数计算距离找位置，但画图时画平滑的线。
        # 这里为了保持定位准确性，建议：
        # 1. 计算距离用 s_info['data'] (原始权重)
        # 2. 画图用 s_data (平滑后)
        # --- 寻找最佳匹配位置 (Localization with Penalty) ---
        raw_s_data = s_info['data']
        best_dist = float('inf')
        best_start_t = -1

        # 遍历所有可能的起始点
        for step_i in range(len(s_penalty)):
            start_t = step_i * s_stride
            if start_t + s_len > x_np.shape[1]: break

            segment = x_np[s_channel, start_t: start_t + s_len]
            # 计算距离用原始数据
            dist = np.mean(np.abs(segment - raw_s_data))
            weighted_dist = dist * s_penalty[step_i]

            if weighted_dist < best_dist:
                best_dist = weighted_dist
                best_start_t = start_t

        if best_start_t != -1:
            # --- 绘图 (画平滑后的曲线) ---
            axs[s_channel].plot(
                np.arange(best_start_t, best_start_t + s_len),
                s_data,  # <--- 这里传入平滑后的数据
                color=colors[rank % len(colors)],
                linewidth=2,
                label=f'Rank-{rank + 1}'
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ==============================================================================
# 3. 数据加载辅助函数
# ==============================================================================

def prepare_test_data(data_path, fold_num, problem):
    """
    使用 dataloader.py 加载数据并返回测试集
    """
    print(f"Loading data via dataloader.py from {data_path} (Fold: {fold_num})...")

    data_path = data_path + "/" + problem
    # 构造 dataloader.load 需要的 config 字典
    config = {
        'data_dir': data_path,
        'fold_num': fold_num
    }
    try:
        # dataloader.load 返回一个包含 'X_test', 'y_test' 等的大字典
        data_dict = load_dataset(config)

        x_test = data_dict['X_test']
        y_test = data_dict['y_test']

        # 转换为 Tensor
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.from_numpy(x_test).float()
            y_test = torch.from_numpy(y_test).long()

        # 维度检查与转置
        # InterpGN 需要 (B, C, T)。如果加载出来是 (B, T, C)，需要转置。
        # 通常 dataloader 处理后的数据可能是 (samples, channels, time) 或 (samples, time, channels)
        # 假设 ADHD 数据通道数约为 22。如果最后一维是 22，说明是 (B, T, C)
        if x_test.shape[-1] < x_test.shape[1] and x_test.shape[-1] == 22:
            print("Transposing input from (B, T, C) to (B, C, T)...")
            x_test = x_test.transpose(1, 2)

        print(f"Data loaded. Test set shape: {x_test.shape}")
        return x_test, y_test

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using dummy data for verification...")
        return None


# ==============================================================================
# 4. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # --- 参数配置 (请根据实际情况调整) ---
    config = {
        # 数据路径 (必须包含 ADHD 文件夹的上一级，或者直接指向 ADHD 文件夹，取决于 dataloader 实现)
        # 根据 user prompt: '../TSCModel/RankSCL/RankSCL/ADHD'
        'data_path': '../Dual_Gate_Model/data',
        'fold_num': 0,  # 默认 fold

        # 模型参数
        'in_channels': 22,  # ADHD 通道数
        'seq_length': 800,  # 序列长度
        'num_classes': 2,  # 类别数
        'num_shapelet': [2, 2, 2, 2],
        'shapelet_len': [0.1, 0.2, 0.3, 0.5],

        'model_path': 'best_model.pth',
        'save_dir': './vis_results',
        'problem': 'VFT'
    }

    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据
    x_test, y_test = prepare_test_data(config['data_path'], config['fold_num'], config['problem'])
    x_test = x_test.to(device)

    # 2. 初始化模型
    print("Initializing model...")
    model = InterpGN(
        in_channels=config['in_channels'],
        seq_length=config['seq_length'],
        num_classes=config['num_classes'],
        num_shapelet=config['num_shapelet'],
        shapelet_len=config['shapelet_len']
    ).to(device)

    # 3. 加载权重
    if os.path.exists(config['model_path']):
        state_dict = torch.load(config['model_path'], map_location=device)
        model.load_state_dict(state_dict)
        print(f"Weights loaded from {config['model_path']}")
    else:
        print(f"Warning: {config['model_path']} not found. Using random weights.")

    # 4. 运行可视化 (画前3个样本)
    print("Starting visualization...")
    samples_to_plot = [0, 1, 2]

    for idx in samples_to_plot:
        if idx >= len(x_test): continue

        save_name = os.path.join(config['save_dir'], f"sample_{idx}_cls{y_test[idx].item()}.png")

        visualize_shapelets_adapted(
            model=model,
            x_data=x_test,
            y_label=y_test,
            sample_id=idx,
            top_k=5,
            save_path=save_name
        )

    print("Done. Check ./vis_results/")