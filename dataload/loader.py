# -*- coding: utf-8 -*-
import os
import os.path as op
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from collections import defaultdict

from dataload.trans import decomposition, vessel_segmentation_normalize, av_classification_normalize


class EyeBallData(Dataset):
    def __init__(self, config, mode):
        super(EyeBallData, self).__init__()

        assert mode == 'train' or mode == 'test' or mode == 'validation'

        Path = op.join(config.path, mode)

        self.mode = mode
        self.patchSize = config.patchSize

        self.imgPath = op.join(Path, 'image')
        self.labPath = op.join(Path, 'label')
        self.vesPath = op.join(Path, 'vessel')

        self.itemList = os.listdir(self.imgPath)
        self.transToTensor = ToTensor()

    def __len__(self):
        return len(self.itemList)

    def __getitem__(self, item):
        img = Image.open(op.join(self.imgPath, self.itemList[item]))
        lab = Image.open(op.join(self.labPath, self.itemList[item]))

        ves = Image.open(op.join(self.vesPath, self.itemList[item])).convert('L')

        originalSize = lab.size
        currentSize = img.size

        lab = torch.Tensor(decomposition(lab))
        ves = self.transToTensor(ves)

        imgVes = vessel_segmentation_normalize(img)
        imgAv = av_classification_normalize(img)

        data = {'imgVes': imgVes, 'labelAv': lab, 'imgAv': imgAv, 'labelVes': ves}

        if self.mode == 'train':
            return data
        elif self.mode == 'validation':
            data['imgVes'] = self.validation_split(imgVes)
            data['imgAv'] = self.validation_split(imgAv)
            return data, originalSize, currentSize, self.itemList[item]
        else:

            point = [0, 45, 90, 135, 180]

            data['imgVes'] = self.test_split(imgVes, point)
            data['imgAv'] = self.test_split(imgAv, point)
            return data, originalSize, currentSize, self.itemList[item]

    def validation_split(self, item: torch.Tensor):

        c, h, w = item.shape
        s = self.patchSize
        imgList = []

        m = w // s + (0 if w % s == 0 else 1)
        n = h // s + (0 if h % s == 0 else 1)

        tmp_w, tmp_h = (m * s - w) // 2, (n * s - h) // 2

        cropItem = torch.zeros((c, n * s, m * s))
        cropItem[:, tmp_h: tmp_h + h, tmp_w: tmp_w + w] = item

        for j in range(int(m)):
            for i in range(int(n)):
                image = cropItem[:, i * s: (i + 1) * s, j * s: (j + 1) * s]
                imgList.append(image)

        return imgList

    def test_split(self, item: torch.Tensor, point: list):

        c, h, w = item.shape
        s = self.patchSize

        m = w // s + 1
        n = h // s + 1

        pos = [(i, j) for i in point for j in point]

        imgDict = dict()

        for idx, p in enumerate(pos):
            newImage = torch.zeros((c,  n * s, m * s))
            newImage[:, p[0]: p[0] + h, p[1]: p[1] + w] = item
            imgDict[p] = newImage

        resDict = defaultdict(list)

        for k in imgDict.keys():
            for j in range(int(m)):
                for i in range(int(n)):
                    image = imgDict[k][:, i * s: (i + 1) * s, j * s: (j + 1) * s]
                    resDict[k].append(image)

        return resDict
