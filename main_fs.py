# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:35:33 2020

@author: yan10
"""

from detector_train import DetectorTrainer
from DSD import DSD
from FeatureExtraction import get_dataset, get_embedded_dim
from model import SSDetector
import pickle
import pretrainedmodels
from ptdec.dec import WDEC
import ptsdae.model as ae
from ptsdae.sdae import StackedDenoisingAutoEncoder
import sys
import time
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from utils import compressed_pickle, decompress_pickle
from wdec import train, DataSetExtract, PotentialScores

start_time = time.time()
if torch.cuda.device_count()>1:
    torch.cuda.set_device(1)
torch.manual_seed(0)

# model_name = 'inceptionresnetv2'
# feature_extractor = pretrainedmodels.__dict__[model_name](
#     num_classes=1000, pretrained='imagenet')
# feature_extractor.eval()
feature_extractor = nn.Identity()

# get dataset and dataloader
print('preparing prerequisites')
data_path     = sys.argv[1] # '../../data/region_proposals'
batch_size    = 400
batch_num     = 1000000
label         = sys.argv[2] # 'bike'
bb_dict_path  = sys.argv[3]
bb_dict       = pickle.load(open(bb_dict_path, 'rb'))
detector_path = 'detector.p' 
print('getting dataset')
ds_train = get_dataset(data_path, label)
ds_train.add_full_supervision(bb_dict)
print('got dataset')
dl = DataLoader(ds_train, batch_size)
# pretrain 
embedded_dim = get_embedded_dim()

MAX_EPOCHS = 35

print('WDEC stage.')
print(f'@ {time.time() - start_time}')

# region classifier we use 3 FC layers (1024,1024,2) with a ReLU
# activation in layers 1-2 and a softmax activation for the output layer
# Dropout is used for the two hidden layers with probability of 0.8
# cross-entropy loss function
detector = SSDetector(feature_extractor, embedded_dim)
cuda = torch.cuda.is_available
if cuda:
    detector.cuda()

# ADAM for optimization with a learning rate of 10?4
det_optimizer = Adam(detector.parameters(), lr=1e-4)
# The learning rate is decreased by a factor of 0.6 every 6 epochs
scheduler = StepLR(det_optimizer, step_size = 6, gamma = 0.6)
def loss(y_hat, y_true):
    return nn.CrossEntropyLoss()(y_hat, y_true.long())

det_trainer = DetectorTrainer(
    model = detector,
    loss_fn = loss,
    optimizer = det_optimizer,
    scheduler = scheduler,
    device = 'cuda' if cuda else 'cpu',
)

for epoch in range(MAX_EPOCHS):
      
    # Train a region classifier with sampled positive and negative regions 
    # get all data needed to compute the potential scores.
    print('\nTraining detector')
    print(f'@ {time.time() - start_time}\n')
    det_trainer.fit(dl, num_epochs = 1, start_epoch = epoch)
    
torch.save(detector.cpu(), detector_path)
# pickle.dump(detector, open(detector_path, 'wb'))
    
    
    
    
    
















