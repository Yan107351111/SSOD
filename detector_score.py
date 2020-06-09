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
from DSD import get_iou

model_name = 'inceptionresnetv2'
feature_extractor = pretrainedmodels.__dict__[model_name](
    num_classes=1000, pretrained='imagenet')
feature_extractor.eval()



def detect(detector, image_path,):
    regions, bounding_boxes = selective_search(image_path, to_file = False)
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
            predictions.append(detector(regions))
    predictions = torch.cat(predictions)
    prediction  = torch.argmax(predictions[:,1])
    return bounding_boxes[prediction]


def evaluate(model, data_path, ground_truth_path, threshold = 0.3):
    images = [i for i in os.listdir(data_path)
              if i.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp',))]   
    IOUs = []    
    bb_dict = pickle.load(open(ground_truth_path, 'rb'))
    for image in images:
        if image in list(bb_dict):
            ground_truth = bb_dict[image]
        else:
            ground_truth = None
        image_path = os.path.join(data_path, image)
        bounding_box = detect(model, image_path)
        if ground_truth is None:
            continue ############## TODO: figure out what to do here
        else: IOUs.append(get_iou(bounding_box, ground_truth)>threshold) 
    
    return torch.mean(torch.stack(IOUs))


if __name__ =='__main__':
    detector_path = sys.argv[1]
    data_path = sys.argv[2]
    ground_truth_path = sys.argv[3]
    
    detector = pickle.load(open(detector_path, 'rb'))
    
    print(evaluate(detector, data_path, ground_truth_path))
    
    

        
    






