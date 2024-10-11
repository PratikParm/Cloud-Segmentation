from torch.nn import Module
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn import MaxPool2d
from torch.nn import ReLU
import torch.nn as nn
import torch
from torchsummary import summary
import time


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=False):
        super(Conv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels//2)
        self.conv2 = nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = self.pooling(res) if not self.bottleneck else res
        return out, res


class UpConv2DBlock(nn.Module):
    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None):
        super(UpConv2DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (last_layer == True and num_classes != None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=in_channels // 2)
        self.conv1 = nn.Conv2d(in_channels=in_channels + res_channels, out_channels=in_channels // 2, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=(3, 3), padding=1)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv2d(in_channels=in_channels // 2, out_channels=num_classes, kernel_size=1)

    def forward(self, x, residual=None):
        out = self.upconv1(x)
        if residual is not None:
            out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer:
            out = self.conv3(out)
        return out


class UNet2D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=None, bottleneck_channel=64):
        super(UNet2D, self).__init__()
        if level_channels is None:
            level_channels = [8, 16, 32]
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv2DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv2DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv2DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv2DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck=True)
        self.s_block3 = UpConv2DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv2DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv2DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    def forward(self, x):
        # Analysis path forward feed
        out, residual_level1 = self.a_block1(x)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        # Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)

        out = torch.sigmoid(out)

        return out


if __name__ == '__main__':
    model = UNet2D(in_channels=3, num_classes=1)
    start_time = time.time()
    summary(model=model, input_size=(3, 256, 256), device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))
