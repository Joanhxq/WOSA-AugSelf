import torch
import torch.nn as nn

from utils import *


class ConvBlock(nn.Module):
    "conv->bn->relu-conv-bn-relu"

    def __init__(self, c_in, c_out, bn=True):
        super(ConvBlock, self).__init__()
        if bn:
            self.func = nn.Sequential(
                nn.Conv2d(c_in,
                          c_out,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True), nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
        else:
            self.func = nn.Sequential(
                nn.Conv2d(c_in,
                          c_out,
                          kernel_size=3,
                          padding=1,
                          stride=1,
                          bias=True), nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True))

        self._init_weight()

    def forward(self, x):
        return self.func(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)


class UpConv(nn.Module):
    "Upsampling"

    def __init__(self, c_in, c_out, up_scale=2):
        super(UpConv, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=up_scale,
                        align_corners=True,
                        mode="bilinear"),
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
        self._init_weight()

    def forward(self, x):
        return self.upsample(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)


class Unet(nn.Module):
    "Unet 2d"

    def __init__(self, args=None, din_level=None, din_skip_level=None):
        super(Unet, self).__init__()
        self.args = args
        self.din_level = din_level
        self.din_skip_level = din_skip_level

        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 1024)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up5 = UpConv(1024, 512)
        self.up_conv5 = ConvBlock(1024, 512)
        self.up4 = UpConv(512, 256)
        self.up_conv4 = ConvBlock(512, 256)
        self.up3 = UpConv(256, 128)
        self.up_conv3 = ConvBlock(256, 128)
        self.up2 = UpConv(128, 64)
        self.up_conv2 = ConvBlock(128, 64)
        self.up_conv1 = nn.Conv2d(64, 2, kernel_size=1)

        self.unet_encoder = [
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5
        ]
        self.unet_up = [self.up5, self.up4, self.up3, self.up2]
        self.unet_decoder = [
            self.up_conv5, self.up_conv4, self.up_conv3, self.up_conv2,
            self.up_conv1
        ]
        self.ps = {1: (266, 134, 266, 134), 2: (132, 68, 132, 68), 3: (66, 34, 66, 34), 4: (32, 18, 32, 18), 5: (16, 9, 16, 9)}

    def get_all_features(self, x):
        states = {}
        for level in range(1, 6):
            x = self.encode(x, states, level)
        for level in range(6, 11):
            x = self.decode(x, states, level)
            if level in self.din_level:
                states[f'dec_{level}'] = x
        return states

    def encode(self, x, states, level, style_states=None):
        assert level in {1, 2, 3, 4, 5}
        if level != 1:
            x = self.maxpool(x)
        x = self.unet_encoder[level - 1](x)
        states[f'enc_{level}'] = x
        if style_states is not None and level in self.din_skip_level:
            if self.args.adain:
                states[f'enc_{level}'] = adain(states[f'enc_{level}'], style_states[f'enc_{level}'])

            if self.args.osa:
                states[f'enc_{level}'] = osa(states[f'enc_{level}'], style_states[f'enc_{level}'])

            if self.args.wosa:
                kh, sh, kw, sw = self.ps[level]
                states[f'enc_{level}'] = wosa_batch(states[f'enc_{level}'], style_states[f'enc_{level}'], kh, kw, sh, sw)

        if style_states is not None and level in self.din_level:
            if self.args.adain:
                x = adain(x, style_states[f'enc_{level}'])

            if self.args.osa:
                x = osa(x, style_states[f'enc_{level}'])

            if self.args.wosa:
                kh, sh, kw, sw = self.ps[level]
                x = wosa_batch(x, style_states[f'enc_{level}'], kh, kw, sh, sw)

        return x

    def decode(self, x, states, level, style_states=None):
        index = {6: 4, 7: 3, 8: 2, 9: 1}
        assert level in {6, 7, 8, 9, 10}
        if level != 10:
            x = self.unet_up[level - 6](x)
            x = torch.cat((x, states[f'enc_{index[level]}']), dim=1)
        x = self.unet_decoder[level - 6](x)
        return x

    def forward(self, x, style=None):
        if style is not None:
            style_states = self.get_all_features(style)
        else:
            style_states = None
        ## =================
        states = {}
        for level in range(1, 6):
            x = self.encode(x, states, level, style_states=style_states)
        for level in range(6, 11):
            x = self.decode(x, states, level, style_states=style_states)
        return x

    def init_weights(self):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
