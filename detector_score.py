#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:38:10 2020

@author: yanivzis@bm
"""

import pickle
import pretrainedmodels
import sys
from SelectiveSearch import selective_search
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

model_name = 'inceptionresnetv2'
feature_extractor = pretrainedmodels.__dict__[model_name](
    num_classes=1000, pretrained='imagenet')
feature_extractor.eval()




detector_path = sys.argv[1]
axu_path = sys.argv[2]



detector = pickle.load(open(detector_path, 'rb'))

def detect(image):
    regions, bounding_boxes = selective_search(image, to_file = False)
    ds = TensorDataset(regions)
    dl = DataLoader(sd, batch_size = 512)
    features = []
    for regions in dl:
        with torch.no_grad():
            features.append(feature_extractor(regions))
    features = torch.cat(features)
    ds = TensorDataset(features)
    dl = DataLoader(sd, batch_size = 512)
    predictions = []
    for feature in dl:
        with torch.no_grad():
            predictions.append(feature_extractor(regions))
    predictions = torch.cat(predictions)
    return bounding_boxes[predictions]


    






