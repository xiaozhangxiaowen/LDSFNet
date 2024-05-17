from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# --------- RGB----------- #
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

# Central Difference Convolution (CDC)
class Conv2d_cdc(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cdc, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

# Spatial attention mechanisms
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

# Texture Feature Enhancement
class TEFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(TEFBlock, self).__init__()
        self.conv_cdc = nn.Sequential(Conv2d_cdc(in_channels, in_channels, kernel_size, stride, padding, dilation, groups, bias, theta),
                                      nn.BatchNorm2d(in_channels),
                                      nn.ReLU(inplace=True),
                                      )

        self.conv_final = nn.Sequential(nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True),
                                        )

        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        b, n, h, w = x.shape

        x_at = F.adaptive_max_pool2d(x, (h // 2, w // 2)) - F.adaptive_avg_pool2d(x, (h // 2, w // 2))
        x_ct = self.conv_cdc(x)

        x = torch.cat([x_at, x_ct], dim=1)
        att = self.sa(x)
        x = x * att + x
        x = self.conv_final(x)

        return x

# SGE
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x


class DWBlock_RGB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DWBlock_RGB, self).__init__()
        self.conv1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.h_switch = nn.Hardswish(inplace=True)

        self.conv2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, 1, dilation, bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.sge = SpatialGroupEnhance(32)

    def forward(self, x):
        # first SeparableConv2d
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.h_switch(out)
        # second SeparableConv2d
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sge(out)

        out = self.h_switch(out)

        return out

# --------- High-frequency----------- #
class SRMConv2d_simple(nn.Module):

    def __init__(self, learnable=True):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel()  # (3,1,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, x):
        x = x[:, 0, :, :] * 0.299 + x[:, 1, :, :] * 0.587 + x[:, 2, :, :] * 0.114
        x = torch.unsqueeze(x, dim=1)

        self.normalized_F()

        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def normalized_F(self):
        central_pixel = (self.kernel.data[:, 0, 2, 2])
        for i in range(3):
            sumed = self.kernel.data[i].sum() - central_pixel[i]
            self.kernel.data[i] /= sumed
            self.kernel.data[i, 0, 2, 2] = -1.0

    def _build_kernel(self):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],
                   [filter2],
                   [filter3]]
        filters = np.array(filters)
        filters = torch.FloatTensor(filters)  # (3, 1, 5, 5)
        return filters

# Multi-Channel SRM Convolution
class MSRMConv(nn.Module):
    def __init__(self, inc, outc):
        super(MSRMConv, self).__init__()
        self.in_planes = inc
        # initialization the weight
        self.truc = nn.Hardtanh(-3, 3)
        const_weight1, const_weight2, const_weight3 = self._build_kernel(inc)  # (inc, 1, 5, 5)

        # get MCSConv kernel
        self.weight1 = nn.Parameter(data=const_weight1, requires_grad=False)
        self.weight2 = nn.Parameter(data=const_weight2, requires_grad=False)
        self.weight3 = nn.Parameter(data=const_weight3, requires_grad=False)

        # self.out_conv = nn.Sequential(
        #     nn.Conv2d(inc, outc, 1, 1, 0, 1, 1, bias=False),
        #     nn.BatchNorm2d(outc),
        #     nn.ReLU(inplace=True)
        # )


    def forward(self, x):
        out1 = F.conv2d(x, self.weight1, stride=1, padding=2, groups=self.in_planes)
        out2 = F.conv2d(x, self.weight2, stride=1, padding=2, groups=self.in_planes)
        out3 = F.conv2d(x, self.weight3, stride=1, padding=2, groups=self.in_planes)

        out = self.truc(out1 + out2 + out3)
        # out = self.out_conv(out)
        # out = out1 + out2 + out3

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters1 = [[filter1]]
        filters2 = [[filter2]]
        filters3 = [[filter3]]

        filters1 = np.array(filters1)
        filters2 = np.array(filters2)
        filters3 = np.array(filters3)

        filters1 = np.repeat(filters1, inc, axis=0)
        filters2 = np.repeat(filters2, inc, axis=0)
        filters3 = np.repeat(filters3, inc, axis=0)

        filters1 = torch.FloatTensor(filters1)  # (inc, 1, 5, 5)
        filters2 = torch.FloatTensor(filters2)  # (inc, 1, 5, 5)
        filters3 = torch.FloatTensor(filters3)  # (inc, 1, 5, 5)
        return filters1, filters2, filters3

# Parameter-free Attention Module
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class DWBlock_Freq(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DWBlock_Freq, self).__init__()
        self.conv1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.h_switch = nn.Hardswish(inplace=True)

        self.simam = simam_module()

        if stride == 2:
            self.dowmsample = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.dowmsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = out + self.simam(out)

        out = out + self.dowmsample(x)
        out = self.h_switch(out)

        return out

# --------- Selective Fusion ----------- #
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SelectiveFusion(nn.Module):
    def __init__(self, channels):
        super(SelectiveFusion, self).__init__()
        self.dwconv = nn.Sequential(SeparableConv2d(channels * 2, channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(channels),
                                    nn.ReLU(inplace=True))

        self.channel_att = ChannelAttention(in_planes=channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc1 = nn.Conv2d(channels, channels // 16, 1, bias=False)
        self.bn_fc1 = nn.BatchNorm2d(channels // 16)
        self.conv_fc2 = nn.Conv2d(channels // 16, 2 * channels, 1, bias=False)

        self.ln = nn.LayerNorm([channels, 1, 1])
        self.relu = nn.ReLU(inplace=True)

        self.D = channels

    def forward(self, x_rgb, x_freq):
        x_fus = torch.cat((x_rgb, x_freq), dim=1)
        x_fus = self.dwconv(x_fus)

        d = self.avg_pool(x_rgb) + self.avg_pool(x_freq)
        d = F.relu(self.bn_fc1(self.conv_fc1(d)))
        d = self.conv_fc2(d)
        d = torch.unsqueeze(d, 1).view(-1, 2, self.D, 1, 1)
        d = F.softmax(d, 1)
        x_rgb = x_rgb * d[:, 0, :, :, :].squeeze(1)
        x_freq = x_freq * d[:, 1, :, :, :].squeeze(1)

        x_fus = x_rgb + x_freq + x_fus
        att = self.channel_att(x_fus)
        x_fus = x_fus * att
        x_fus = self.relu(self.ln(self.avg_pool(x_fus)))

        return x_fus

if __name__ == "__main__":
    a = torch.rand((32, 256, 16, 16))
    b = torch.rand((32, 256, 16, 16))
    m = SelectiveFusion(256)
    out = m(a, b)
    print(out.shape)
    pass
