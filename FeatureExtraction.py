# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:33:57 2020

@author: yan10
"""

import pretrainedmodels
import sys
import torch
import torchvision
import torchvision.transforms as T

if __name__ == '__main__':
    data_path = sys.argv[1] #'..\data\region_proposals'
    if len(sys.argv)>2:
        batch_size = sys.argv[2]
    else: batch_size = 5
    
    model_name = 'inceptionresnetv2'
    model = pretrainedmodels.__dict__[model_name](
        num_classes=1000, pretrained='imagenet')
    model.eval()
    
    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.ImageFolder(
        root      = data_path,
        transform = transform,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2)
    
    inputs, labels = next(iter(train_dataloader))
    for label in labels:
        print(label)

