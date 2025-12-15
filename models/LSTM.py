import torch
import torch.nn as nn


class CommonLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=4):
        super(CommonLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        # batch_first=True 表示输入格式为 (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0  # 防止过拟合
        )

        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 原始输入 x: (Batch, 8, 426) -> (Batch, Features, Time)

        # LSTM 需要 (Batch, Time, Features)，所以交换维度 1 和 2
        x = x.permute(0, 2, 1)  # 变为 (Batch, 426, 8)

        # LSTM 前向传播
        # out: 包含所有时间步的输出 (Batch, 426, 128)
        # (h_n, c_n): 最后一个时间步的隐状态和细胞状态
        out, (h_n, c_n) = self.lstm(x)

        # 我们取最后一个时间步的输出作为分类特征
        # out[:, -1, :] 形状为 (Batch, 128)
        last_time_step_out = out[:, -1, :]

        # 全连接分类
        out = self.fc(last_time_step_out)  # (Batch, 4)
        return out