# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Net.structure import ConvolutionPart, CotMlpBlock


class BaseUnet(nn.Module):
    """
    Unet:
        default: 16, 32, 64, 128, 256
    """

    def __init__(self, inputFeature, outputFeature, numFeature=16, activation=None, mode='single'):
        super().__init__()

        n = numFeature
        self.mode = mode

        filters = [inputFeature]
        for i in range(5):
            filters.append((2 ** i) * n)

        self.EncodeBlock = nn.ModuleList()
        self.DecodeBlock = nn.ModuleList()

        for i in range(len(filters) - 2):
            if mode == 'start' or mode == 'single' or i == 0:
                self.EncodeBlock.append(ConvolutionPart([filters[i], filters[i + 1]], layers_size=2))
            else:
                self.EncodeBlock.append(ConvolutionPart([filters[i] + filters[i + 1], filters[i + 1]], layers_size=2))

        for i in range(1, len(filters) - 1):
            tmp = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv2d(filters[-i], filters[-i - 1], kernel_size=1),
                                nn.ReLU())
            self.DecodeBlock.append(tmp)
            self.DecodeBlock.append(ConvolutionPart([filters[-i], filters[-i - 1]], layers_size=2))

        self.MaxPool = nn.MaxPool2d(2)
        self.middleConv = ConvolutionPart([filters[-2], filters[-1]], layers_size=2)
        self.finalConv = nn.Conv2d(n, outputFeature, kernel_size=3, stride=1, padding=1)

        if activation is not None:
            self.isActivation = True
            if activation == 'softmax':
                self.activation = nn.Softmax(dim=1)
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            else:
                raise
        else:
            self.isActivation = False

    def forward(self, x, aboveFeatures=None):
        features = []

        for i in range(len(self.EncodeBlock)):
            if self.mode == 'start' or self.mode == 'single' or i == 0:
                x = self.EncodeBlock[i](x)
            else:
                x = self.EncodeBlock[i](torch.cat([x, aboveFeatures[-i]], dim=1))
            features.append(x)
            x = self.MaxPool(x)

        x = self.middleConv(x)

        nextFeatures = []
        for i in range(0, len(self.DecodeBlock), 2):
            x = self.DecodeBlock[i](x)
            x = self.DecodeBlock[i + 1](torch.cat([features[-i // 2 - 1], x], dim=1))
            nextFeatures.append(x)
        nextFeatures.pop()

        res = self.activation(self.finalConv(x)) if self.isActivation else self.finalConv(x)

        if self.mode == 'end' or self.mode == 'single':
            return res
        else:
            return res, nextFeatures


class CirUnet(nn.Module):
    def __init__(self, inputFeature, outputFeature, interFeature, K, numFeature=16):
        super(CirUnet, self).__init__()

        self.layers = nn.ModuleList([BaseUnet(inputFeature=inputFeature,
                                              outputFeature=interFeature, numFeature=numFeature, mode='start')])
        self.ves = nn.ModuleList([nn.Conv2d(interFeature, outputFeature, 1)])

        for _ in range(K - 2):
            self.layers.append(
                BaseUnet(inputFeature=inputFeature + interFeature,
                         outputFeature=interFeature, numFeature=numFeature, mode='middle'))
            self.ves.append(nn.Conv2d(interFeature, outputFeature, 1))

        self.layers.append(
            BaseUnet(inputFeature=inputFeature + interFeature,
                     outputFeature=interFeature, numFeature=numFeature, mode='end'))
        self.ves.append(nn.Conv2d(interFeature, outputFeature, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = []
        feature, aboveFeatures = None, None

        for idx, (layer, ves) in enumerate(zip(self.layers, self.ves)):
            if idx == 0:
                feature, aboveFeatures = layer(img)
                out.append(ves(feature))
            elif idx < len(self.layers) - 1:
                feature, aboveFeatures = layer(torch.cat((img, feature), dim=1), aboveFeatures)
                out.append(ves(feature))
            else:
                feature = layer(torch.cat((img, feature), dim=1), aboveFeatures)
                out.append(ves(feature))

        return out


class Trans(nn.Module):

    def __init__(self, inputFeature, outputFeature, numFeature, mode='start'):
        super().__init__()

        self.mode = mode

        n = numFeature
        filters = [inputFeature]
        for i in range(5):
            filters.append((2 ** i) * n)

        self.EncodeBlock = nn.ModuleList()
        self.DecodeBlock = nn.ModuleList()

        for i in range(len(filters) - 2):
            if mode == 'start' or i == 0:
                tmp = nn.Sequential(
                    ConvolutionPart([filters[i], filters[i + 1]], layers_size=1),
                    CotMlpBlock(filters[i + 1])
                )
            else:
                tmp = nn.Sequential(
                    ConvolutionPart([filters[i] + filters[i + 1], filters[i + 1]], layers_size=1),
                    CotMlpBlock(filters[i + 1])
                )
            self.EncodeBlock.append(tmp)

        for i in range(1, len(filters) - 1):
            tmp = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv2d(filters[-i], filters[-i - 1], kernel_size=1),
                                nn.ReLU())
            self.DecodeBlock.append(tmp)
            self.DecodeBlock.append(ConvolutionPart([filters[-i], filters[-i - 1]], layers_size=2))

        self.MaxPool = nn.MaxPool2d(2)

        self.middleConv = ConvolutionPart([filters[-2], filters[-1]])
        self.finalConv = nn.Conv2d(n, outputFeature, kernel_size=3, stride=1, padding=1)

    def forward(self, x, aboveFeatures=None):
        features = []

        for i in range(len(self.EncodeBlock)):
            if self.mode == 'start' or i == 0:
                x = self.EncodeBlock[i](x)
            else:
                x = self.EncodeBlock[i](torch.cat([x, aboveFeatures[-i]], dim=1))
            features.append(x)
            x = self.MaxPool(x)

        x = self.middleConv(x)

        nextFeatures = []
        for i in range(0, len(self.DecodeBlock), 2):
            x = self.DecodeBlock[i](x)
            x = self.DecodeBlock[i + 1](torch.cat([features[-i // 2 - 1], x], dim=1))
            nextFeatures.append(x)
        nextFeatures.pop()

        if self.mode == 'end':
            return self.finalConv(x)
        else:
            return self.finalConv(x), nextFeatures


class ConnectTrans(nn.Module):
    def __init__(self, inputFeature, outputFeature, interFeature=None, depth=2):
        super(ConnectTrans, self).__init__()
        interFeature = interFeature or outputFeature

        self.layers = nn.ModuleList(
            [Trans(inputFeature=inputFeature, outputFeature=interFeature, numFeature=16, mode='start')]
        )
        self.outLayers = nn.ModuleList([nn.Conv2d(interFeature, outputFeature, 1)])
        for _ in range(depth - 2):
            self.layers.append(
                Trans(inputFeature=inputFeature + interFeature, outputFeature=interFeature, numFeature=16, mode='middle')
            )
            self.outLayers.append(nn.Conv2d(interFeature, outputFeature, 1))

        self.layers.append(
            Trans(inputFeature=inputFeature + interFeature, outputFeature=interFeature, numFeature=16, mode='end')
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, fil):
        res = x
        out, aboveFeatures = [], None

        for idx, layer in enumerate(self.layers):

            if idx == 0:
                res, aboveFeatures = layer(torch.cat((x, fil), dim=1))
                out.append(self.outLayers[0](res))

            elif idx < len(self.outLayers):
                res, aboveFeatures = layer(torch.cat((x, fil, res), dim=1), aboveFeatures)
                out.append(self.outLayers[idx](res))

            else:
                res = layer(torch.cat((x, fil, res), dim=1), aboveFeatures)
                out.append(res)

        return out


class FFModule(nn.Module):
    def __init__(self, alpha):
        super(FFModule, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        y = self.alpha * (torch.exp(-torch.abs(x - 0.5)) - torch.exp(torch.tensor(-1 / 2))) + 1
        return y * x


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == '__main__':
    # net = Trans(4, 4, 4, 32)
    TransNet = ConnectTrans(4, 4, 16, 4)
    CirNet = CirUnet(2, 1, 2, 4)
    # net = CirUnet(4, 1, 4, 4, 16)
    test(CirNet)

    inputAvFeatures = torch.rand(2, 3, 256, 256)
    inputVesFeatures = torch.rand(2, 2, 256, 256)
    filFeatures = torch.rand(2, 1, 256, 256)
    outputAvFeatures = TransNet(inputAvFeatures, filFeatures)
    outputVesFeatures = CirNet(inputVesFeatures)
    print(outputAvFeatures[-1].shape)
    print(outputVesFeatures[-1].shape)
