import torch
import torch.nn as nn

# ImageNet pretrained weight for DarkNet-19
model_urls = {
    "darknet19": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet19.pth",
}


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet19(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d(2, 2)
        )

        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d(2, 2)
        )

        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d(2, 2)
        )

        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )
        self.maxpool_4 = nn.MaxPool2d(2, 2)

        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1)
        )
        self.maxpool_5 = nn.MaxPool2d(2, 2)

        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        c1 = self.conv_1(x)  # [B, 32, H/2, W/2]
        c2 = self.conv_2(c1)  # [B, 64, H/4, W/4]
        c3 = self.conv_3(c2)  # [B, 128, H/8, W/8]
        c3 = self.conv_4(c3)  # [B, 256, H/8, W/8]
        c4 = self.conv_5(self.maxpool_4(c3))  # [B, 512, H/16, W/16]
        c5 = self.conv_6(self.maxpool_5(c4))  # [B, 1024, H/32, W/32]

        return c5


def build_backbone(model_name='darknet19', pretrained=False):
    if model_name == 'darknet19':
        model = DarkNet19()
        feat_dim = 1024

    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['darknet19']
        # checkpoint state dict
        checkpoint_state_dict = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # model state dict
        model_state_dict = model.state_dict()
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

        model.load_state_dict(checkpoint_state_dict)

    return model, feat_dim
