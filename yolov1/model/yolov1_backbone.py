import torch
import torch.nn as nn
import torchvision.models as models


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.shortcut(identity)
        x = self.relu2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += self.shortcut(identity)
        x = self.relu3(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks=layers[3], stride=2)

        self._init_weights(zero_init_residual)

    def _init_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # [B, C, H/2, W/2]
        x = self.bn1(x)  # [B, C, H/2, W/2]
        x = self.relu1(x)  # [B, C, H/2, W/2]
        x = self.maxpool(x)  # [B, C, H/4, W/4]

        x = self.layer1(x)  # [B, C, H/4, W/4]
        x = self.layer2(x)  # [B, C, H/8, W/8]
        x = self.layer3(x)  # [B, C, H/16, W/16]
        x = self.layer4(x)  # [B, C, H/32, W/32]

        return x


def build_backbone(model_name='resnet18', pretrained=True):
    if model_name == 'resnet18':
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
            model = nn.Sequential(*list(model.children())[:-2])
        else:
            model = ResNet(ResidualBlock, [2, 2, 2, 2], True)
        feat_dim = 512
    elif model_name == 'resnet34':
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            model = models.resnet34(weights=weights)
            model = nn.Sequential(*list(model.children())[:-2])
        else:
            model = ResNet(ResidualBlock, [3, 4, 6, 3], True)
        feat_dim = 512
    elif model_name == 'resnet50':
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
            model = nn.Sequential(*list(model.children())[:-2])
        else:
            model = ResNet(Bottleneck, [3, 4, 6, 3], True)
        feat_dim = 2048
    elif model_name == 'resnet101':
        if pretrained:
            weights = models.ResNet101_Weights.IMAGENET1K_V1
            model = models.resnet101(weights=weights)
            model = nn.Sequential(*list(model.children())[:-2])
        else:
            model = ResNet(Bottleneck, [3, 4, 23, 3], True)
        feat_dim = 2048

    return model, feat_dim
