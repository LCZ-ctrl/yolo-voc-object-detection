import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolov4_backbone import Conv_BN_SiLU, CSPBlock


class Yolov4PaFPN(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256, width=1.0, depth=1.0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        c3, c4, c5 = in_channels

        # top down
        ## P5 -> P4
        self.reduce_layer_1 = Conv_BN_SiLU(c5, int(512 * width), kernel_size=1)
        self.top_down_layer_1 = CSPBlock(in_channels=c4 + int(512 * width),
                                         out_channels=int(512 * width),
                                         expand_ratio=0.5,
                                         nblocks=int(3 * depth),
                                         shortcut=False)

        ## P4 -> P3
        self.reduce_layer_2 = Conv_BN_SiLU(c4, int(256 * width), kernel_size=1)
        self.top_down_layer_2 = CSPBlock(in_channels=c3 + int(256 * width),
                                         out_channels=int(256 * width),
                                         expand_ratio=0.5,
                                         nblocks=int(3 * depth),
                                         shortcut=False)

        # bottom up
        ## P3 -> P4
        self.reduce_layer_3 = Conv_BN_SiLU(int(256 * width), int(256 * width), kernel_size=3, padding=1, stride=2)
        self.bottom_up_layer_1 = CSPBlock(in_channels=int(256 * width) + int(256 * width),
                                          out_channels=int(512 * width),
                                          expand_ratio=0.5,
                                          nblocks=int(3 * depth),
                                          shortcut=False)

        ## P4 -> P5
        self.reduce_layer_4 = Conv_BN_SiLU(int(512 * width), int(512 * width), kernel_size=3, padding=1, stride=2)
        self.bottom_up_layer_2 = CSPBlock(in_channels=int(512 * width) + int(512 * width),
                                          out_channels=int(1024 * width),
                                          expand_ratio=0.5,
                                          nblocks=int(3 * depth),
                                          shortcut=False)

        # output proj layers
        if out_channels is not None:
            self.out_layers = nn.ModuleList([
                Conv_BN_SiLU(in_channels, out_channels, kernel_size=1)
                for in_channels in [int(256 * width), int(512 * width), int(1024 * width)]
            ])
            self.out_channels = [out_channels] * 3
        else:
            self.out_layers = None
            self.out_channels = [int(256 * width), int(512 * width), int(1024 * width)]

    def forward(self, features):
        # c3: [B, 256, H/8, W/8]
        # c4: [B, 512, H/16, W/16]
        # c5: [B, 1024, H/32, W/32]
        c3, c4, c5 = features

        c6 = self.reduce_layer_1(c5)  # [B, 512, H/32, W/32]
        c7 = F.interpolate(c6, scale_factor=2.0)  # s32->s16, [B, 512, H/16, W/16]
        c8 = torch.cat([c7, c4], dim=1)  # [B, 1024, H/16, W/16]
        c9 = self.top_down_layer_1(c8)  # [B, 512, H/16, W/16]
        # P3/8
        c10 = self.reduce_layer_2(c9)  # [B, 256, H/16, W/16]
        c11 = F.interpolate(c10, scale_factor=2.0)  # s16->s8, [B, 256, H/8, W/8]
        c12 = torch.cat([c11, c3], dim=1)  # [B, 512, H/8, W/8]
        c13 = self.top_down_layer_2(c12)  # [B, 256, H/8, W/8]
        # p4/16
        c14 = self.reduce_layer_3(c13)  # [B, 256, H/16, W/16]
        c15 = torch.cat([c14, c10], dim=1)  # [B, 512, H/16, W/16]
        c16 = self.bottom_up_layer_1(c15)  # [B, 512, H/16, W/16]
        # p5/32
        c17 = self.reduce_layer_4(c16)  # [B, 512, H/32, W/32]
        c18 = torch.cat([c17, c6], dim=1)  # [B, 1024, H/32, W/32]
        c19 = self.bottom_up_layer_2(c18)  # [B, 1024, H/32, W/32]

        out_feats = [c13, c16, c19]  # [P3, P4, P5]

        # output proj layers
        if self.out_layers is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(in_channels, out_channels=None):
    fpn_net = Yolov4PaFPN(in_channels, out_channels, width=1.0, depth=1.0)
    return fpn_net
