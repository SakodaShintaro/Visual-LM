import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2dWithBatchNorm, self).__init__()
        self.conv_ = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, padding=kernel_size // 2)
        self.norm_ = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, kernel_size, reduction):
        super(ResidualBlock, self).__init__()
        self.conv_and_norm0_ = Conv2dWithBatchNorm(channel_num, channel_num, kernel_size)
        self.conv_and_norm1_ = Conv2dWithBatchNorm(channel_num, channel_num, kernel_size)
        self.linear0_ = nn.Linear(channel_num, channel_num // reduction, bias=False)
        self.linear1_ = nn.Linear(channel_num // reduction, channel_num, bias=False)

    def forward(self, x):
        t = x
        t = self.conv_and_norm0_.forward(t)
        t = F.relu(t)
        t = self.conv_and_norm1_.forward(t)

        y = F.avg_pool2d(t, [t.shape[2], t.shape[3]])
        y = y.view([-1, t.shape[1]])
        y = self.linear0_.forward(y)
        y = F.relu(y)
        y = self.linear1_.forward(y)
        y = torch.sigmoid(y)
        y = y.view([-1, t.shape[1], 1, 1])
        t = t * y

        t = F.relu(x + t)
        return t


class CNNModel(nn.Module):
    def __init__(self, input_channel_num, hidden_size=128):
        super(CNNModel, self).__init__()
        self.first_conv = Conv2dWithBatchNorm(in_channels=input_channel_num, out_channels=hidden_size, kernel_size=3)
        self.blocks = nn.Sequential()
        block_num = 3
        for i in range(block_num):
            self.blocks.add_module(f"block{i}", ResidualBlock(hidden_size, kernel_size=3, reduction=8))
        self.final_conv = Conv2dWithBatchNorm(in_channels=hidden_size, out_channels=1, kernel_size=3)

    def forward(self, x):
        x = self.first_conv(x)
        short_cut1 = x
        x = F.relu(x)
        x = self.blocks(x)
        x = x + short_cut1
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x
