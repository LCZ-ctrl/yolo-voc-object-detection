import torch
import torch.nn as nn
from .yolov4_backbone import Conv_BN_SiLU


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, pooling_size=5):
        super().__init__()

        hidden_channels = int(in_channels * expand_ratio)
        self.out_dim = out_channels

        self.cv1 = Conv_BN_SiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = Conv_BN_SiLU(hidden_channels * 4, out_channels, kernel_size=1)
        self.m = nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=pooling_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        x = torch.cat([x, y1, y2, y3], dim=1)
        x = self.cv2(x)
        return x


class SPPFBlockCSP(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, pooling_size=5):
        super().__init__()

        hidden_channels = int(in_channels * expand_ratio)
        self.out_dim = out_channels
        self.cv1 = Conv_BN_SiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = Conv_BN_SiLU(in_channels, hidden_channels, kernel_size=1)
        self.m = nn.Sequential(
            Conv_BN_SiLU(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            SPPF(hidden_channels, hidden_channels, expand_ratio=1.0, pooling_size=pooling_size),
            Conv_BN_SiLU(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        )
        self.cv3 = Conv_BN_SiLU(hidden_channels * 2, self.out_dim, kernel_size=1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y


def build_neck(model, in_channels, out_channels):
    if model == 'sppf':
        neck = SPPF(in_channels, out_channels, expand_ratio=0.5, pooling_size=5)
    elif model == 'csp_sppf':
        neck = SPPFBlockCSP(in_channels, out_channels, expand_ratio=0.5, pooling_size=5)

    return neck
