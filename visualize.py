import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def transform_array(arr):
    """
    对一个NumPy数组进行变换：先取绝对值，然后将范围缩放到 [0, 1]。

    参数:
    arr (np.ndarray): 输入的NumPy数组。

    返回:
    np.ndarray: 变换后的数组。
    """
    # 1. 取绝对值
    arr_abs = np.abs(arr)

    # 2. Min-Max Scaling
    min_val = arr_abs.min()
    max_val = arr_abs.max()

    # 处理所有元素都相同的特殊情况，避免除以零
    if max_val == min_val:
        # 返回一个形状相同、所有元素为0的数组
        return arr

    normalized_arr = (arr_abs - min_val) / (max_val - min_val)

    return normalized_arr

def get_position_channel_maps(model):
    """
    从InterpGN模型中提取所有 position_channel_map。

    Args:
        model (InterpGN): 训练好的或未训练的InterpGN模型。

    Returns:
        list: 一个包含元组的列表，每个元组是 (map_data, title)。
              map_data 是一个2D numpy数组。
              title 是为图表准备的标题。
    """
    maps = []
    # 遍历SBM中的每个Shapelet模块
    for i, shapelet_module in enumerate(model.sbm.shapelets):
        # 获取map，从计算图中分离，并转移到CPU，转换为numpy数组
        map_data = shapelet_module.position_channel_map.detach().cpu().numpy()

        # 获取该shapelet的长度信息，用于标题
        shapelet_length = (i+1)*10

        # 创建一个有意义的标题
        title = f'Shapelet Block {i+1} (Length: {shapelet_length}%)'

        maps.append((map_data, title))
    return maps


# # [修改] visualize_and_save_map 函数
# def visualize_and_save_map(map_data: np.ndarray, title: str, save_path: str):
#     """
#     [修改] 将单个 position_channel_map 可视化为热力图并保存到文件。
#     移除了色块间的缝隙，并保持了对刻度的控制。
#     """
#
#     if map_data is None or map_data.size == 0:
#         print(f"Skipping visualization for '{title}' due to empty data.")
#         return
#     map_data = transform_array(map_data)
#
#     plt.figure(figsize=(12, 6))
#
#     # 保持对刻度频率的控制，您可以根据需要调整这些值
#     x_tick_frequency = 25
#     y_tick_frequency = 2
#
#     ax = sns.heatmap(
#         map_data,
#         xticklabels=x_tick_frequency,
#         yticklabels=y_tick_frequency,
#         cmap='Blues_r',
#         # [修改] 将 linewidths 参数移除，以消除色块间的缝隙
#         # linewidths=.5,  <-- 这一行被删除了
#         cbar_kws={'label': 'Importance Score (Lower is More Important)'}
#     )
#
#     ax.set_title(title, fontsize=16)
#     ax.set_xlabel("Subsequence Position Index", fontsize=12)
#     ax.set_ylabel("Channel Index", fontsize=12)
#
#     # 让x轴的标签旋转，防止它们太长而重叠
#     plt.xticks(rotation=45)
#
#     plt.savefig(save_path, bbox_inches='tight', dpi=150)
#     plt.close()
#
#     print(f"Saved visualization to: {save_path}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def visualize_and_save_map(map_data: np.ndarray, title: str, save_path: str):
    """
    将单个 position_channel_map 可视化为热力图并保存到文件
    颜色映射：
        0.0 = RGB(97, 143, 198)  深蓝色
        0.5 = RGB(255, 255, 255) 纯白色
        1.0 = RGB(255, 156, 156) 淡红色
    """
    if map_data is None or map_data.size == 0:
        print(f"Skipping visualization for '{title}' due to empty data.")
        return

    map_data = transform_array(map_data)

    plt.figure(figsize=(12, 6))

    # 刻度频率设置
    x_tick_frequency = 25
    y_tick_frequency = 2

    # 创建自定义颜色映射
    cdict = {
        'red': [(0.0, 97 / 255, 97 / 255),
                (0.5, 1.0, 1.0),
                (1.0, 255 / 255, 255 / 255)],

        'green': [(0.0, 143 / 255, 143 / 255),
                  (0.5, 1.0, 1.0),
                  (1.0, 156 / 255, 156 / 255)],

        'blue': [(0.0, 198 / 255, 198 / 255),
                 (0.5, 1.0, 1.0),
                 (1.0, 156 / 255, 156 / 255)]
    }
    custom_cmap = LinearSegmentedColormap('BlueWhiteRed', cdict)

    # 使用自定义颜色映射
    ax = sns.heatmap(
        map_data,
        xticklabels=x_tick_frequency,
        yticklabels=y_tick_frequency,
        cmap=custom_cmap,
        vmin=0,
        vmax=1,
        #cbar_kws={'label': 'Importance Score (Lower is More Important)'}
    )

    # 1. 获取 Color Bar 的 Axes 对象
    # ax.figure.axes 包含了图中的所有子图，最后一个通常是 color bar
    cbar_ax = ax.figure.axes[-1]

    # 2. 修改 Color Bar 刻度标签的字体大小
    cbar_ax.tick_params(labelsize=16)  # 你可以调整这里的数值

    # 3. (推荐) 使用 Axes 对象的方法来设置 Color Bar 的标题
    # 这样做可以让你同时控制标题的字体大小
    #cbar_ax.set_ylabel('Importance Score (Lower is More Important)', fontsize=16)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Channel", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(rotation=45)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 提高DPI以获得更清晰的图像
    plt.close()

    print(f"Saved visualization to: {save_path}")