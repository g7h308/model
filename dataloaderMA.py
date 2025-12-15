import torch
import pandas as pd
import numpy as np
import scipy.io as scio
import os
import glob

def UFFT_subject_data(data_path, subject=1):
    """
    从 UFFT 数据集中加载指定被试的完整试次（trial）数据，不进行滑窗切分。

    Args:
        data_path (str): UFFT 数据集的根目录路径。
        subject (int): 要加载的被试编号。

    Returns:
        tuple: 包含两个 NumPy 数组的元组 (data, label)
        - data (np.ndarray): 形状为 (样本数, 通道数, 时间点数) 的数据数组。
          样本数 = sheet的数量, 通道数 = sheet的列数, 时间点数 = sheet的行数。
        - label (np.ndarray): 形状为 (样本数,) 的标签数组。
    """
    # 构建数据和标签文件的路径
    data_file = f'{data_path}/{subject}/{subject}.xls'
    desc_file = f'{data_path}/{subject}/{subject}_desc.xls'

    # --- 数据加载 ---
    # 使用 pandas 一次性读取所有 sheet，返回一个字典 {sheet_name: DataFrame}
    # sheet_name=None 是关键，header=None 表示没有表头
    all_sheets_dict = pd.read_excel(data_file, header=None, sheet_name=None)

    # 将所有 sheet 的数据 (DataFrame) 转换为 NumPy 数组，并放入一个列表中
    # all_sheets_dict.values() 会返回所有 DataFrame
    # 假设每个 sheet 的形状是 (时间点数, 通道数)
    all_trials_list = [sheet.values for sheet in all_sheets_dict.values()]

    # 将列表转换为一个3D NumPy数组
    # 此时的形状是 (样本数, 时间点数, 通道数)
    data = np.array(all_trials_list)

    # 进行维度转置，以满足 (样本数, 通道数, 时间点数) 的要求
    # 交换第1和第2个维度（从0开始计数）
    data = data.transpose((0, 2, 1))

    # --- 标签加载 ---
    # 读取描述文件
    desc_df = pd.read_excel(desc_file, header=None)

    # 提取第一列作为标签，并减1（通常标签从1开始，模型需要从0开始）
    # .values 将 DataFrame 转换为 NumPy 数组，[:, 0] 选择第一列
    label = desc_df.values[:, 0] - 1

    print(f'被试 {subject} 的数据加载完成。')
    # print(f'Data shape: {data.shape}')
    # print(f'Label shape: {label.shape}')

    return data, label


# def MA_subject_data(path, sub):
#     """
#     load MA data.
#
#     Args:
#         path: Data path of the MA dataset.
#         sub: Index of subject.
#     """
#     data = []
#     label = []
#
#     # read label
#     file_path = os.path.join(path, str(sub), str(sub)+'_desc.mat')
#     signal_label = np.array(scio.loadmat(file_path)['label']).squeeze()
#     for k in range(len(signal_label)):
#         if signal_label[k] == 1:
#             signal_label[k] = 0
#         elif signal_label[k] == 2:
#             signal_label[k] = 1
#
#     # read data (60, 72, 30); (9, 19) -> [-2, 10]s
#     for wins in range(9, 19):
#         file_path = os.path.join(path, str(sub), str(wins) + '_oxy.mat')
#         oxy = np.array(scio.loadmat(file_path)['signal']).transpose((2, 1, 0))[:, :, :30]
#         file_path = os.path.join(path, str(sub), str(wins) + '_deoxy.mat')
#         deoxy = np.array(scio.loadmat(file_path)['signal']).transpose((2, 1, 0))[:, :, :30]
#         # (60, 72, 30)
#         hb = np.concatenate((oxy, deoxy), axis=1)
#
#         data.append(hb)
#         label.append(signal_label)
#
#     print(str(sub) + '  OK')
#     data = np.array(data).transpose((1, 0, 2, 3))
#     label = np.array(label).transpose((1, 0))
#     # print(data.shape)
#     # print(label.shape)
#     return data, label

import pandas as pd
import numpy as np


def MA_subject_data(data_path):
    """
    加载数据集 B，并保留每个时间序列的原始完整长度。

    Args:
        data_path (str): 数据集路径。

    Returns:
        feature : fNIRS 信号数据，形状为 (num_samples, 72, time_points)。
        label : fNIRS 标签。
    """
    feature = []
    label = []
    for sub in range(1, 30):
        # --- 代码前面部分保持不变 ---
        name = data_path + '/' + str(sub) + '/' + str(sub) + '_oxy.xls'
        oxy = pd.read_excel(name, header=None, sheet_name=None)
        name = data_path + '/' + str(sub) + '/' + str(sub) + '_deoxy.xls'
        deoxy = pd.read_excel(name, header=None, sheet_name=None)
        name = data_path + '/' + str(sub) + '/' + str(sub) + '_desc.xls'
        desc = pd.read_excel(name, header=None)

        HbO = []
        HbR = []
        for i in range(1, 61):
            name = 'Sheet' + str(i)
            HbO.append(oxy[name].values)
            HbR.append(deoxy[name].values)

        HbO = np.array(HbO).transpose((0, 2, 1))
        HbR = np.array(HbR).transpose((0, 2, 1))
        desc = np.array(desc)

        # --- 修改开始 ---
        # 动态获取时间点的数量 (原始长度)
        time_points = HbO.shape[2]

        HbO_MA = []
        HbO_BL = []
        HbR_MA = []
        HbR_BL = []
        for i in range(60):
            # 移除 [start:end] 切片，保留完整数据
            if desc[i, 0] == 1:
                HbO_MA.append(HbO[i, :, :])
                HbR_MA.append(HbR[i, :, :])
            elif desc[i, 0] == 2:
                HbO_BL.append(HbO[i, :, :])
                HbR_BL.append(HbR[i, :, :])

        # 使用动态获取的 time_points 变量进行 reshape
        HbO_MA = np.array(HbO_MA).reshape((30, 1, 36, time_points))
        HbO_BL = np.array(HbO_BL).reshape((30, 1, 36, time_points))
        HbR_MA = np.array(HbR_MA).reshape((30, 1, 36, time_points))
        HbR_BL = np.array(HbR_BL).reshape((30, 1, 36, time_points))
        # --- 修改结束 ---

        HbO_MA = np.concatenate((HbO_MA, HbR_MA), axis=1)
        HbO_BL = np.concatenate((HbO_BL, HbR_BL), axis=1)

        for i in range(30):
            feature.append(HbO_MA[i, :, :, :])
            feature.append(HbO_BL[i, :, :, :])
            label.append(0)
            label.append(1)

        print(str(sub) + '  OK')

    feature = np.array(feature)
    label = np.array(label)

    # 最后的 reshape 操作，将 (..., 2, 36, time_points) 转换为 (..., 72, time_points)
    # 原始形状: (1740, 2, 36, 原始时间点数)
    # 目标形状: (1740, 72, 原始时间点数)
    # 使用 -1 可以让 numpy 自动计算维度，代码更健壮
    num_samples = feature.shape[0]
    num_time_points = feature.shape[3]
    feature = feature.reshape(num_samples, -1, num_time_points)  # -1 会自动计算为 2 * 36 = 72

    print('feature ', feature.shape)
    print('label ', label.shape)

    return feature, label


def KFold_train_test_set(sub_data, label, data_index, test_index, n_fold):
    train_index = np.setdiff1d(data_index, test_index[n_fold])
    X_train = sub_data[train_index]
    y_train = label[train_index]
    X_test = sub_data[test_index[n_fold]]
    y_test = label[test_index[n_fold]]

    return X_train, y_train, X_test, y_test


def LOSO_train_test_set(all_data, all_label, n_sub, task_id):
    if task_id == 0:
        all_sub = 30  # UFFT
    elif task_id == 1:
        all_sub = 29  # MA

    sub_index = [np.arange(all_sub)]
    train_index = np.setdiff1d(sub_index, n_sub)
    X_train = all_data[train_index]
    y_train = all_label[train_index]
    X_test = all_data[n_sub]
    y_test = all_label[n_sub]
    Sub, N, D, C, S = X_train.shape
    X_train = X_train.reshape((Sub * N, D, C, S))
    y_train = y_train.reshape((Sub * N))
    return X_train, y_train, X_test, y_test


def load_all_data(data_path, task_id):
    """
    load the UFFT or MA dataset.

    Args:
        data_path: Data path of the UFFT or MA dataset.
        task_id: Specify task. '0' is UFFT and '1' is MA.
    """
    all_data = []
    all_label = []
    if task_id == 0:
        all_sub = 30  # UFFT
    elif task_id == 1:
        all_sub = 29  # MA

    for n_sub in range(1, all_sub + 1):
        if task_id == 0:
            sub_data, sub_label = UFFT_subject_data(data_path, subject=n_sub)
        elif task_id == 1:
            sub_data, sub_label = MA_subject_data(path=data_path, sub=n_sub)

        T, W, C, S = sub_data.shape
        sub_data = sub_data.reshape((T * W, 1, C, S))
        sub_label = sub_label.reshape((T * W))
        all_data.append(sub_data)
        all_label.append(sub_label)

    all_data = np.array(all_data)
    all_label = np.array(all_label)
    # print(all_data.shape)
    # print(all_label.shape)
    return all_data, all_label


class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature, label, transform=True):
        self.feature = feature
        self.label = label
        self.transform = transform
        self.feature = torch.tensor(self.feature, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        if self.transform:
            mean, std = self.feature[item].mean(), self.feature[item].std()
            self.feature[item] = (self.feature[item] - mean) / std

        return self.feature[item], self.label[item]


def load_subjects_raw_data(data_dir):
    """
    读取文件夹下所有 CSV，按受试者组织数据。
    假设所有 CSV 中提取出的 chunk 长度都是一致的。
    """
    # 按文件名排序，保证顺序固定
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    subjects_data = []

    print(f"正在读取 {len(csv_files)} 个受试者文件...")

    for file_name in csv_files:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path)

        # 假设前8列是特征，chunk是第9列，label是第10列
        feature_cols = df.columns[:8]
        chunk_col = 'chunk'
        label_col = 'label'

        grouped = df.groupby(chunk_col)

        sub_samples = []
        sub_labels = []

        for _, group in grouped:
            # 提取特征: (Time_Steps, 8)
            feat = group[feature_cols].values.astype(np.float32)
            lbl = int(group[label_col].iloc[0])

            # 转为 Tensor
            sub_samples.append(torch.from_numpy(feat))
            sub_labels.append(lbl)

        subjects_data.append({
            'subject_id': file_name,
            'samples': sub_samples,  # List[Tensor]
            'labels': sub_labels  # List[int]
        })

    print("所有受试者数据读取完毕。")
    return subjects_data


# ================= 2. 核心处理 (针对定长数据优化) =================

def get_kfold_data(subjects_data, fold_num, k=5, target_labels=None):
    """
    1. 按受试者划分 Train/Test
    2. 筛选指定标签 (如 [0, 2])
    3. 自动将标签重新映射为 0, 1, 2...
    4. 将 List 堆叠为 Tensor (因为长度一致)
    5. 标准化 (基于训练集)
    """

    # --- A. 划分受试者 ---
    total_subjects = len(subjects_data)
    fold_size = total_subjects // k
    indices = np.arange(total_subjects)

    start = fold_num * fold_size
    end = (fold_num + 1) * fold_size if fold_num != k - 1 else total_subjects

    test_idx = indices[start:end]
    train_idx = np.concatenate([indices[:start], indices[end:]])

    print(f"Fold {fold_num}: 训练受试者 {len(train_idx)} 人, 测试受试者 {len(test_idx)} 人")

    train_subjects = [subjects_data[i] for i in train_idx]
    test_subjects = [subjects_data[i] for i in test_idx]

    # --- B. 筛选数据 & 标签映射准备 ---
    if target_labels is not None:
        target_labels = sorted(list(set(target_labels)))  # 比如 [0, 2]
        # 创建映射字典: 0->0, 2->1
        label_map = {old_label: new_idx for new_idx, old_label in enumerate(target_labels)}
        print(f"标签映射规则: {label_map}")
    else:
        label_map = None  # 不筛选，也不映射（或者你需要自己处理）

    def extract_and_stack(subject_list):
        X_list = []
        y_list = []

        for sub in subject_list:
            raw_samples = sub['samples']
            raw_labels = sub['labels']

            for i in range(len(raw_samples)):
                lbl = raw_labels[i]

                # 如果指定了标签，且当前标签不在目标中，跳过
                if target_labels is not None and lbl not in target_labels:
                    continue

                X_list.append(raw_samples[i])

                # 处理标签映射
                if label_map is not None:
                    y_list.append(label_map[lbl])
                else:
                    y_list.append(lbl)

        # === 关键步骤：因为长度一致，直接 stack 成一个大 Tensor ===
        # 结果形状: (Total_Samples, Time_Steps, Channels)
        if len(X_list) > 0:
            X_tensor = torch.stack(X_list)
            y_tensor = torch.tensor(y_list, dtype=torch.long)
        else:
            # 防止空数据报错
            X_tensor = torch.empty(0)
            y_tensor = torch.empty(0)

        return X_tensor, y_tensor

    # 获取堆叠后的 Tensor
    train_X, train_y = extract_and_stack(train_subjects)
    test_X, test_y = extract_and_stack(test_subjects)

    print(f"筛选后数据形状: Train X: {train_X.shape}, Test X: {test_X.shape}")

    if len(train_X) == 0:
        raise ValueError("训练集为空，请检查 target_labels 是否正确。")

    # --- C. 标准化 (仅使用 Train 计算 Mean/Std) ---
    # train_X shape: (N, T, C)
    # 我们需要在 N 和 T 维度上求均值，保留 C (通道) 维度
    mean = train_X.mean(dim=(0, 1))
    std = train_X.std(dim=(0, 1))
    std[std == 0] = 1.0  # 避免除零

    print(f"标准化参数 (Mean): {mean.numpy()}")

    # 应用标准化 (利用广播机制)
    # (N, T, C) - (C,) -> OK
    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    train_X = train_X.permute(0, 2, 1)
    test_X = test_X.permute(0, 2, 1)

    return train_X, train_y, test_X, test_y


def load_process_and_split(folder_path, fold_num, n_splits=5):
    """
    读取数据 -> 形状转置 -> 时间维度标准化 -> K折划分

    返回形状: (N, 8, 426)
    """

    # --- 1. 读取文件 ---
    file_pattern = os.path.join(folder_path, "*.csv")
    csv_files = sorted(glob.glob(file_pattern))

    if len(csv_files) == 0:
        raise ValueError("未找到CSV文件")

    print(f"正在处理 {len(csv_files)} 个文件...")

    ROWS_PER_SAMPLE = 426
    SAMPLES_PER_FILE = 16

    all_X = []
    all_y = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        for i in range(SAMPLES_PER_FILE):
            start = i * ROWS_PER_SAMPLE
            end = (i + 1) * ROWS_PER_SAMPLE

            # 取特征 (426, 8)
            sample_data = df.iloc[start:end, :-1].values
            # 取标签
            sample_label = df.iloc[start, -1]

            all_X.append(sample_data)
            all_y.append(sample_label)

    X = np.array(all_X)  # (1088, 426, 8)
    y = np.array(all_y)  # (1088,)

    target_labels = [0, 2]
    mask = np.isin(y, target_labels)

    # 2. 使用这个掩码来过滤 X 和 y
    X = X[mask]
    y = y[mask]

    # --- 2. 维度转置 (N, L, C) -> (N, C, L) ---
    # 变为 (1088, 8, 426)
    X = X.transpose(0, 2, 1)

    # --- 3. 时间维度标准化 (Time-wise Standardization) ---
    # 目的：让每个样本的每个通道，在时间轴上均值为0，标准差为1
    # axis=2 代表时间维度 (426那一维)

    # 计算均值和标准差，keepdims=True 保持形状为 (1088, 8, 1) 以便广播
    mean = np.mean(X, axis=2, keepdims=True)
    std = np.std(X, axis=2, keepdims=True)

    # 为了防止除以0，加上一个极小值 epsilon
    epsilon = 1e-8

    # 执行标准化
    X = (X - mean) / (std + epsilon)

    print(f"标准化完成。X mean: {X.mean():.4f}, X std: {X.std():.4f} (全局近似)")

    # --- 4. 五折划分 ---
    total_samples = len(X)
    fold_size = total_samples / n_splits

    test_start = int(fold_num * fold_size)
    test_end = int((fold_num + 1) * fold_size)

    indices = np.arange(total_samples)
    test_idx = indices[test_start:test_end]
    train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Fold {fold_num} 划分完成:")
    print(f"Train X: {X_train.shape}, Test X: {X_test.shape}")

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()

    X_test= torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    return X_train, y_train, X_test, y_test


# ================= 使用示例 =================

if __name__ == "__main__":
    # 假设你的文件夹路径是 './data'
    folder_path = r"./fNIRS2MW/whole_data"

    # 获取第一折 (fold_num=0, 前20%做测试集)
    try:
        # 这里为了演示，假设你有这个文件夹，实际运行时替换为你的真实路径
        X_train, y_train, X_test, y_test = load_process_and_split(folder_path, fold_num=0)

        sample_0_ch_0 = X_train[0, 0, :]  # 取出 (426,)
        print(sample_0_ch_0)
        print(f"\n验证: 样本0-通道0 -> 均值: {sample_0_ch_0.mean():.6f}, 标准差: {sample_0_ch_0.std():.6f}")
    except Exception as e:
        print(e)