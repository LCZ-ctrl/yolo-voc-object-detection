import torch
import torch.nn as nn


class DecoupledHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=20):
        super().__init__()

        # cls branch
        self.cls_out_dim = max(out_channels, num_classes)
        cls_layers = [nn.Conv2d(in_channels, self.cls_out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(self.cls_out_dim),
                      nn.LeakyReLU(0.1, inplace=True),

                      nn.Conv2d(self.cls_out_dim, self.cls_out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(self.cls_out_dim),
                      nn.LeakyReLU(0.1, inplace=True)]
        self.cls_feats = nn.Sequential(*cls_layers)

        # reg branch
        self.reg_out_dim = max(out_channels, 64)
        reg_layers = [nn.Conv2d(in_channels, self.reg_out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(self.reg_out_dim),
                      nn.LeakyReLU(0.1, inplace=True),

                      nn.Conv2d(self.reg_out_dim, self.reg_out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(self.reg_out_dim),
                      nn.LeakyReLU(0.1, inplace=True)]
        self.reg_feats = nn.Sequential(*reg_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)
        return cls_feats, reg_feats


def build_head(in_channels, out_channels, num_classes=20):
    head = DecoupledHead(in_channels, out_channels, num_classes)
    return head
