# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


class MultiLoss(nn.Module):
    def __init__(self, config, flag, sign, ep=1e-6):
        super(MultiLoss, self).__init__()
        self.ep = ep

        avWeight = torch.FloatTensor([1, 1, 3, 5]).to(config.device)
        self.avCriterion = nn.CrossEntropyLoss(weight=avWeight).to(config.device)

        self.vesselCriterionFinal = nn.BCELoss().to(config.device)
        self.vesselCriterion = nn.BCEWithLogitsLoss().to(config.device)

        self.flag, self.sign = flag, sign

    def dice_loss(self, predVessel, labelVessel):
        intersection = 2 * torch.sum(predVessel * labelVessel) + self.ep
        union = torch.sum(predVessel) + torch.sum(labelVessel) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, predAv, predVessel, labelAv, labelVessel):
        avLoss = vesselLoss = 0

        if self.flag:
            for idx, pred in enumerate(predAv):
                avLoss += 1 / (len(predAv) - idx) * self.avCriterion(pred, labelAv)
        else:
            avLoss += self.avCriterion(predAv, labelAv)

        if self.sign:
            for idx, pred in enumerate(predVessel):
                if idx == len(predVessel) - 1:
                    vesselLoss += self.vesselCriterionFinal(pred, labelVessel)
                    vesselLoss += self.dice_loss(pred, labelVessel)
                else:
                    vesselLoss += 1 / (len(predVessel) - idx) * self.vesselCriterion(pred, labelVessel)
                    vesselLoss += 0.1 * 1 / (len(predVessel) - idx) * self.dice_loss(torch.sigmoid(pred), labelVessel)
        else:
            vesselLoss += self.vesselCriterion(predVessel, labelVessel)
            vesselLoss += 0.1 * self.dice_loss(torch.sigmoid(predVessel), labelVessel)

        return vesselLoss + avLoss
