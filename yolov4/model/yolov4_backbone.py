import torch
import torch.nn as nn

# ImageNet pretrained weight
model_urls = {
    "cspdarknet53": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet53_silu.pth",
}


class Conv_BN_SiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, shortcut=False):
        super().__init__()

        hidden_channels = int(out_channels * expand_ratio)
        self.cv1 = Conv_BN_SiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = Conv_BN_SiLU(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        h = self.cv2(self.cv1(x))
        return x + h if self.shortcut else h


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, nblocks=1, shortcut=False):
        super().__init__()

        hidden_channels = int(out_channels * expand_ratio)
        self.cv1 = Conv_BN_SiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = Conv_BN_SiLU(in_channels, hidden_channels, kernel_size=1)
        self.m = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, expand_ratio=1.0, shortcut=shortcut)
            for _ in range(nblocks)
        ])
        self.cv3 = Conv_BN_SiLU(2 * hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x1)
        out = self.cv3(torch.cat([x3, x2], dim=1))

        return out


class CSPDarkNet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.feat_dims = [256, 512, 1024]

        # P1
        self.layer_1 = nn.Sequential(
            Conv_BN_SiLU(3, 32, kernel_size=3, padding=1),
            Conv_BN_SiLU(32, 64, kernel_size=3, padding=1, stride=2),
            CSPBlock(64, 64, expand_ratio=0.5, nblocks=1, shortcut=True)
        )

        # P2
        self.layer_2 = nn.Sequential(
            Conv_BN_SiLU(64, 128, kernel_size=3, padding=1, stride=2),
            CSPBlock(128, 128, expand_ratio=0.5, nblocks=2, shortcut=True)
        )

        # P3
        self.layer_3 = nn.Sequential(
            Conv_BN_SiLU(128, 256, kernel_size=3, padding=1, stride=2),
            CSPBlock(256, 256, expand_ratio=0.5, nblocks=8, shortcut=True)
        )

        # P4
        self.layer_4 = nn.Sequential(
            Conv_BN_SiLU(256, 512, kernel_size=3, padding=1, stride=2),
            CSPBlock(512, 512, expand_ratio=0.5, nblocks=8, shortcut=True)
        )

        # P5
        self.layer_5 = nn.Sequential(
            Conv_BN_SiLU(512, 1024, kernel_size=3, padding=1, stride=2),
            CSPBlock(1024, 1024, expand_ratio=0.5, nblocks=4, shortcut=True)
        )

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]
        return outputs


def build_backbone(model_name='cspdarknet53', pretrained=False):
    if model_name == 'cspdarknet53':
        backbone = CSPDarkNet53()
        feat_dims = backbone.feat_dims

    if pretrained:
        url = model_urls[model_name]
        if url is not None:
            print('Loading pretrained weight ...')
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = backbone.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                else:
                    checkpoint_state_dict.pop(k)
                    # print(k)

            backbone.load_state_dict(checkpoint_state_dict)
        else:
            print('No backbone pretrained: CSPDarkNet53')

    return backbone, feat_dims
