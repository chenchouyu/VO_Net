# -*- coding: utf-8 -*-

import PIL.ImageFilter

import numpy as np
import torch

import cv2
from torchvision.transforms import ToTensor


def vessel_segmentation_normalize(img):

    _, imgG, _ = img.split()
    imgG = imgG.filter(PIL.ImageFilter.MedianFilter(3))
    imgG = imgG.filter(PIL.ImageFilter.SHARPEN)
    imgG = imgG.filter(PIL.ImageFilter.EDGE_ENHANCE_MORE)

    img = np.array(img)
    _, g, _ = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    tmp = clahe.apply(g)
    tmp.astype('uint8')

    ch1 = ToTensor()(imgG)
    ch2 = ToTensor()(tmp)

    return torch.cat([ch1, ch2], dim=0).type(torch.float)


def av_classification_normalize(img):

    new_img = np.array(img, dtype=np.uint8)
    return ToTensor()(new_img.astype('uint8'))


def decomposition(label_av):
    # label_av: given PIL Image
    label_av = np.copy(np.asarray(label_av))
    label = np.zeros_like(label_av[..., 0])

    label[label_av[:, :, 0] == 255] = 3
    label[label_av[:, :, 2] == 255] = 2
    label[label_av[:, :, 1] == 255] = 1

    return label
