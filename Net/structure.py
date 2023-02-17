# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class ConvolutionPart(nn.Module):
    def __init__(self, channels, kernels=3, strides=1, layers_size=1, isRelu=True):
        """
        :param channels: Including the input channel, marked with channels[0].
        :param kernels: Items used to compute the parameter "padding".
        :param strides: Items used to compute the parameter "padding".
        :param layers_size: Numbers of layers of module.

        Warning: Make sure that "len(channels) == layers_size + 1" and "len(kernels) == layers_size".
        """
        super().__init__()

        if isinstance(kernels, int):
            kernels = [kernels] * layers_size
        if isinstance(strides, int):
            strides = [strides] * layers_size

        assert type(channels) == list, \
            'Something wrong with the type of the parameter "channels" or the length of it. Please check it.'
        assert type(kernels) == list and len(kernels) == layers_size, \
            'Something wrong with the type of the parameter "kernels" or the length of it. Please check it.'
        assert type(strides) == list and len(strides) == layers_size, \
            'Something wrong with the type of the parameter "strides" or the length of it. Please check it.'

        layers = []

        for i in range(layers_size):
            if i >= len(channels) - 1:
                layers.append(nn.Conv2d(channels[-1], channels[-1], kernel_size=kernels[i],
                                        stride=strides[i], padding=(kernels[i] - 1) // 2))
                layers.append(nn.BatchNorm2d(channels[-1]))
            else:
                layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernels[i],
                                        stride=strides[i], padding=(kernels[i] - 1) // 2))
                layers.append(nn.BatchNorm2d(channels[i + 1]))

            if isRelu:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        return self.layers(X)


class Attention(nn.Module):

    def __init__(self, dim, head_dim=64, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class ConvMlpBlock(nn.Module):
    def __init__(self, inputFeature, outputFeature):
        super(ConvMlpBlock, self).__init__()

        self.Conv1 = ConvolutionPart([inputFeature, inputFeature], kernels=1)
        self.Conv2 = ConvolutionPart([inputFeature, inputFeature], kernels=3)
        self.Conv3 = ConvolutionPart([inputFeature, inputFeature], kernels=1)

        self.Mlp = Mlp(inputFeature)
        self.finalConv = ConvolutionPart([inputFeature, outputFeature], 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        residual = X
        out = self.Conv1(X)
        out = self.Conv2(out)
        out = self.Conv3(out)
        out = self.Mlp(out + residual)
        out = self.finalConv(out)
        return out


class AttentionMlpBlock(nn.Module):
    def __init__(self, inputFeature):
        super(AttentionMlpBlock, self).__init__()
        self.Attention = Attention(inputFeature)
        self.Mlp = Mlp(inputFeature)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        X = self.Attention(X)
        X = self.Mlp(X)
        return X


class Mlp(nn.Module):
    def __init__(self, inputFeature, hiddenFeature=None,
                 outputFeature=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        outputFeature = outputFeature or inputFeature
        hiddenFeature = hiddenFeature or inputFeature
        self.fc1 = nn.Conv2d(inputFeature, hiddenFeature, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hiddenFeature, outputFeature, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Cot(nn.Module):
    def __init__(self, featureNum, kernel_size=3):
        super(Cot, self).__init__()
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            nn.Conv2d(featureNum, featureNum, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(featureNum),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(featureNum, featureNum, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(featureNum)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * featureNum, 2 * featureNum // factor, 1, bias=False),
            nn.BatchNorm2d(2 * featureNum // factor),
            nn.ReLU(),
            nn.Conv2d(2 * featureNum // factor, kernel_size * kernel_size * featureNum, 1, stride=1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).view(bs, c, -1)
        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


class CotMlpBlock(nn.Module):
    def __init__(self, inputFeature):
        super(CotMlpBlock, self).__init__()
        self.Attention = Cot(inputFeature)

    def forward(self, X):
        X = self.Attention(X)
        return X
