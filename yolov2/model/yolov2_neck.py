import torch
import torch.nn as nn


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, pooling_size=5):
        super().__init__()

        hidden_channels = int(in_channels * expand_ratio)
        self.out_dim = out_channels

        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=pooling_size // 2)

        self.cv2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)

        x = torch.cat([x, y1, y2, y3], dim=1)
        x = self.cv2(x)
        return x


def build_neck(in_channels, out_channels):
    neck = SPPF(in_channels, out_channels, 0.5, 5)
    return neck
