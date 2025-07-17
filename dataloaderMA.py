import torch
import pandas as pd
import numpy as np
import scipy.io as scio
import os

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

