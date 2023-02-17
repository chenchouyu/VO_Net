# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init

from Net.block import ConnectTrans, CirUnet, BaseUnet, FFModule
from Net.structure import ConvolutionPart


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class VONet(nn.Module):
    def __init__(self,
                 inputFeature: int,
                 outputFeature: int,
                 MS: bool = True,
                 CA: bool = True,
                 FF: bool = True,
                 D: int = 4,
                 K: int = 4,
                 alpha: float = 1.0
                 ) -> object:

        super(VONet, self).__init__()

        self.alpha = alpha

        self.Gf = ConnectTrans(inputFeature=inputFeature,
                               outputFeature=4,
                               interFeature=16,
                               depth=D) if CA else BaseUnet(inputFeature=inputFeature,
                                                            outputFeature=16,
                                                            numFeature=64)

        self.Gc = CirUnet(inputFeature=2,
                          outputFeature=1,
                          interFeature=2,
                          K=K) if MS else BaseUnet(inputFeature=2,
                                                   outputFeature=1,
                                                   numFeature=64)

        self.finalAvConv = ConvolutionPart([16, outputFeature], kernels=1, isRelu=False)

        if FF:
            self.FF = FFModule(alpha)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.isFF = FF
        self.isMS = MS
        self.isCA = CA

        self.apply(_weights_init)

    def forward(self, imgAv, imgVes):

        outVesselList = self.Gc(imgVes)

        if self.isMS:
            outVessel = self.sigmoid(outVesselList[-1])
        else:
            outVessel = self.sigmoid(outVesselList)

        if self.isCA:
            outAvList = self.Gf(imgAv, outVessel)
            outAv = outAvList[-1]
        else:
            outAvList = self.Gf(torch.cat([imgAv, outVessel], dim=1))
            outAv = outAvList

        if self.isFF:
            outVessel = self.FF(outVessel)
            outAv = self.finalAvConv(outAv * outVessel)
        else:
            outAv = self.finalAvConv(outAv)

        if self.isMS:
            outVesselList[-1] = outVessel
        else:
            outVesselList = outVessel

        if self.isCA:
            outAvList[-1] = self.softmax(outAv)
        else:
            outAvList = self.softmax(outAv)

        return outAvList, outVesselList


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == '__main__':
    # net = Trans(4, 4, 4, 32)
    Net = VONet(4, 4, CA=False)
    # net = CirUnet(4, 1, 4, 4, 16)
    test(Net)

    inputAvFeatures = torch.rand(2, 3, 256, 256)
    inputVesFeatures = torch.rand(2, 2, 256, 256)
    filFeatures = torch.rand(2, 1, 256, 256)
    outputAvFeatures, outputVesFeatures = Net(inputAvFeatures, inputVesFeatures)
    print(outputAvFeatures[-1].shape)
    print(outputVesFeatures[-1].shape)