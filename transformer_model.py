import torch
import torch.nn as nn
import torch.nn.functional as F

from constant import IMAGE_HEIGHT, IMAGE_WIDTH


class Conv2dWithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2dWithBatchNorm, self).__init__()
        self.conv_ = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, padding=kernel_size // 2)
        self.norm_ = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t


class TransformerModel(nn.Module):
    def __init__(self, input_channel_num, hidden_size=32):
        super(TransformerModel, self).__init__()

        # ネットワークのパラメータ
        dim_feedforward = 128
        memory_length = 16

        # 一番最初のconv
        self.first_conv = Conv2dWithBatchNorm(in_channels=input_channel_num, out_channels=hidden_size, kernel_size=3)

        # 最初に入力-メモリでCross Attentionしてメモリを更新
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, nhead=8, dim_feedforward=dim_feedforward)
        self.first_cross_attention = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # メモリ内部でSelf Attention
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=8, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # メモリと入力をマージ
        self.merge_conv = Conv2dWithBatchNorm(in_channels=hidden_size + memory_length * hidden_size, out_channels=hidden_size, kernel_size=1)

        # 1ch化
        self.final_conv = Conv2dWithBatchNorm(in_channels=hidden_size, out_channels=1, kernel_size=3)

        # メモリ
        self.memory = nn.Parameter(torch.randn(memory_length, hidden_size))

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(IMAGE_HEIGHT * IMAGE_WIDTH, 1, hidden_size))

    def forward(self, x):
        raw_input = x
        x = self.first_conv(x)
        short_cut1 = x
        x = x.flatten(2)
        x = x.permute([2, 0, 1])
        x = x + self.positional_encoding
        memory = self.memory.repeat(x.shape[1], 1, 1)
        memory = memory.permute([1, 0, 2])
        memory = self.first_cross_attention(memory, x)
        memory = self.transformer_encoder(memory)
        memory = memory.permute([1, 0, 2])
        memory = memory.flatten(1)
        memory = memory.view([memory.shape[0], memory.shape[1], 1, 1])
        memory = memory.repeat(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        x = torch.cat([short_cut1, memory], 1)
        x = self.merge_conv(x)
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x + raw_input
