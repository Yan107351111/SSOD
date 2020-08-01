# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:43:52 2020

@author: yan10
"""
import torch
from torch import nn

class SSDetector(nn.Module):
    def __init__(self, embedded_dim):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(embedded_dim, 1024))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.8))
        layers.append(nn.Linear(1024, 1024))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.8))
        layers.append(nn.Linear(1024, 2))
        
        self.classifier = nn.Sequential(*layers) 
        
        self.softmax = nn.Softmax(-1)
        self.SMTemp  = 1.
        self._activate = True

        
    def forward(self, x):
        
        y = self.classifier(x)
        
        if self._activate:
            y = self.softmax(y/self.SMTemp)
        
        return y
