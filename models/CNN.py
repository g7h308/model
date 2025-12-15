import torch
import torch.nn as nn


class Common1DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Common1DCNN, self).__init__()

        # 第一层卷积块
        # 输入: (Batch, 8, 426) -> 输出: (Batch, 32, 213)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第二层卷积块
        # 输入: (Batch, 32, 213) -> 输出: (Batch, 64, 106)
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第三层卷积块
        # 输入: (Batch, 64, 106) -> 输出: (Batch, 128, 53)
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 自适应平均池化
        # 将时间维度(53) 压缩为 1，变为 (Batch, 128, 1)
        # 这样做的好处是无论前面输入长度怎么变，进入全连接层的特征数固定为 128
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接分类层
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (Batch, 8, 426)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.adaptive_pool(x)  # (Batch, 128, 1)
        x = x.view(x.size(0), -1)  # 展平为 (Batch, 128)

        out = self.fc(x)  # (Batch, 4)
        return out