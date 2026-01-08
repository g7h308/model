import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
import warnings
import logging

warnings.filterwarnings('ignore')

# 配置 Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_stats_channel_wise(train_data):
    """
    仅根据训练数据计算均值和标准差。
    假设 train_data 形状为 (Samples, Time, Channels)
    我们希望对每个 Channel 进行标准化，因此跨 Samples(0) 和 Time(1) 聚合。
    """
    # 均值 shape: (Channels,)
    mean = np.mean(train_data, axis=(0, 1))
    # 标准差 shape: (Channels,)
    std = np.std(train_data, axis=(0, 1))

    # 防止除以0，将极小的std替换为1
    std[std < 1e-8] = 1.0

    return mean, std


def apply_normalization(data, mean, std):
    """
    应用标准化: (Data - Mean) / Std
    利用广播机制 (Broadcasting) 自动处理维度。
    Data: (N, T, C), Mean: (C,), Std: (C,)
    """
    return (data - mean) / std


def load(config):
    Data = {}
    fold_num = config['fold_num']
    # 确保 fold_num 是 int
    if fold_num is None:
        raise ValueError("Config 中必须包含 'fold_num'")
    fold_num = int(fold_num)

    problem = config['data_dir'].split('/')[-1]
    # 文件名建议加上 fold 信息，避免不同 fold 覆盖同一文件
    npy_path = os.path.join(config['data_dir'], f"{problem}_n2_fold{fold_num}.npy")

    # --- 1. 检查是否存在预处理文件 ---
    if os.path.exists(npy_path):
        logger.info(f"Loading preprocessed data from {npy_path} ...")
        Data_npy = np.load(npy_path, allow_pickle=True)
        item = Data_npy.item()

        Data['max_len'] = item.get('max_len')
        Data['X_train'] = item.get('X_train')
        Data['y_train'] = item.get('y_train')
        Data['X_val'] = item.get('X_val')
        Data['y_val'] = item.get('y_val')
        Data['X_test'] = item.get('X_test')
        Data['y_test'] = item.get('y_test')

        # 兼容性加载（如果存在全量训练数据）
        Data['All_train_data'] = item.get('All_train_data')
        Data['All_train_label'] = item.get('All_train_label')

        logger.info(f"Loaded: Train({len(Data['y_train'])}), Val({len(Data['y_val'])}), Test({len(Data['y_test'])})")

    else:
        # --- 2. 加载原始 Excel 数据 ---
        logger.info("Loading and preprocessing data from Excel files ...")

        # 参数设置
        n = 2  # 扩增倍数/降采样因子
        adhd_dir = os.path.join(config['data_dir'], "ADHD")
        hc_dir = os.path.join(config['data_dir'], "HC")

        # 读取文件列表
        adhd_files = sorted([os.path.join(adhd_dir, f) for f in os.listdir(adhd_dir) if f.endswith('.xlsx')])
        hc_files = sorted([os.path.join(hc_dir, f) for f in os.listdir(hc_dir) if f.endswith('.xlsx')])

        dataframes = []
        labels = []
        group_ids = []  # 用于 GroupKFold，防止同一人数据泄露

        # 加载 ADHD (Label=1)
        for idx, file in enumerate(adhd_files):
            df = pd.read_excel(file)
            dataframes.append(df.values)  # 转为numpy array
            labels.append(1)
            group_ids.append(idx)  # 唯一ID

        # 加载 HC (Label=0)
        # Group ID 继续累加，不与 ADHD 重复
        start_id = len(adhd_files)
        for idx, file in enumerate(hc_files):
            df = pd.read_excel(file)
            dataframes.append(df.values)
            labels.append(0)
            group_ids.append(start_id + idx)

        # 转换为 numpy 格式
        # 假设所有 excel 形状相同，或者这里需要 padding (目前代码假设形状一致)
        feature_raw = np.array(dataframes)
        # 当前 shape: (Samples, Time, Features)

        labels = np.array(labels)
        group_ids = np.array(group_ids)

        logger.info(f"Raw feature shape: {feature_raw.shape}")
        n_samples, n_time, n_features = feature_raw.shape

        # --- 3. 数据扩增 / 切分 ---
        # 调整时间长度以适应 n
        n_time_adjusted = (n_time // n) * n

        split_features = []
        split_labels = []
        split_groups = []

        for i in range(n):
            # 这里的逻辑是降采样 (Downsampling)
            # 例如 n=2, 取 t=0,2,4... 和 t=1,3,5...
            indices = np.arange(i, n_time_adjusted, n)

            sub_data = feature_raw[:, indices, :]
            split_features.append(sub_data)

            # 标签和组ID 复制一份
            split_labels.append(labels)
            split_groups.append(group_ids)

        # 合并扩增后的数据
        feature = np.concatenate(split_features, axis=0)  # (Samples*n, Time/n, Features)
        label = np.concatenate(split_labels, axis=0)
        groups = np.concatenate(split_groups, axis=0)

        logger.info(f"Augmented feature shape: {feature.shape}")
        Data['max_len'] = feature.shape[1]  # 记录时间维度长度

        # --- 4. 数据集划分 (Split) ---
        # 此时 feature shape: (N, T, C)

        # Step A: 划分 Train+Val (80%) 和 Test (20%)
        # random_state 固定为 42，确保不同 Fold 下，测试集是同一个，保证公平
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(gss.split(feature, label, groups))

        X_train_val = feature[train_val_idx]
        y_train_val = label[train_val_idx]
        groups_train_val = groups[train_val_idx]

        X_test = feature[test_idx]
        y_test = label[test_idx]

        # Step B: 对 Train+Val 进行 5折交叉验证，划分出 Train 和 Val
        # 注意：这里不需要 random_state 固定，GroupKFold 是确定性的
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(X_train_val, y_train_val, groups_train_val))

        if fold_num >= 5:
            raise ValueError("fold_num cannot exceed 4 for 5-fold CV")

        train_idx, val_idx = folds[fold_num]

        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]

        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]

        logger.info(f"Fold {fold_num} Split:")
        logger.info(f"  Train: {X_train.shape[0]} (Groups: {len(np.unique(groups_train_val[train_idx]))})")
        logger.info(f"  Val:   {X_val.shape[0]} (Groups: {len(np.unique(groups_train_val[val_idx]))})")
        logger.info(f"  Test:  {X_test.shape[0]}")

        # --- 5. 标准化 (Normalization) ---
        # 【重要】仅在 X_train 上计算均值和方差，避免数据泄露
        mean_vec, std_vec = compute_stats_channel_wise(X_train)

        # 应用到所有数据集
        X_train = apply_normalization(X_train, mean_vec, std_vec)
        X_val = apply_normalization(X_val, mean_vec, std_vec)
        X_test = apply_normalization(X_test, mean_vec, std_vec)

        # --- 6. 维度转换 (Transpose) ---
        # 从 (N, T, C) -> (N, C, T) 以适应大多数深度学习模型 (如 1D-CNN)
        X_train = X_train.transpose(0, 2, 1)
        X_val = X_val.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)

        # --- 7. 组装与保存 ---
        Data['X_train'] = X_train
        Data['y_train'] = y_train
        Data['X_val'] = X_val
        Data['y_val'] = y_val
        Data['X_test'] = X_test
        Data['y_test'] = y_test

        # 如果需要合并的训练集 (用于某些特殊的后续处理，通常不需要)
        Data['All_train_data'] = np.concatenate([X_train, X_val], axis=0)
        Data['All_train_label'] = np.concatenate([y_train, y_val], axis=0)

        # 保存为 .npy
        np.save(npy_path, Data, allow_pickle=True)
        logger.info(f"Data saved to {npy_path}")

    return Data