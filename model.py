# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:43:52 2020

@author: yan10
"""
import torch
from torch import nn

class SSDetector(nn.Module):
    def __init__(self,feature_extractor, embedded_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.llayer0 = nn.Linear(embedded_dim, 1024)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.8)
        self.llayer1 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.8)
        self.llayer2 = nn.Linear(1024, 2)
        self.softmax = nn.Softmax()

        
    def farward(self, x):
        with torch.no_grad():
            y = self.feature_extractor(x)
        y = self.llayer0(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.llayer1(y)
        y = self.dropout(y)
        y = self.llayer2(y)
        y = self.softmax(y)
        
        return y
