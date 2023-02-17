# -*- coding: utf-8 -*-

import argparse
import os

from test import test
from train import train
from utils import get_config, get_model, get_test_model
# from tools.prepocess import preprocess
from tools.process import Process

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--mode', default='test')
    arg.add_argument('--preprocess', default=False)
    arg.add_argument('--jsonName', default='./config.json')
    arg = arg.parse_args()

    config, preConfig = get_config(arg.jsonName, arg.mode)
    p = Process(preConfig)

    if arg.mode == 'train':
        if arg.preprocess:
            p.run_train()
        net, modelName = get_model(arg.jsonName)
        train(config, net, modelName)

    else:
        p.run_test()
        testModelNames = ['Unet_Unet_None',
                          'Unet_Unet_FF',
                          'Unet_CA_None',
                          'Unet_CA_FF',
                          'MS_Unet_None',
                          'MS_Unet_FF',
                          'MS_CA_None',
                          'MS_CA_FF']

        for modelName in testModelNames:
            if modelName not in os.listdir('./model'):
                continue
            else:
                net = get_test_model(modelName)
                test(config, net, modelName)

