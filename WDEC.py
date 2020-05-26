# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:35:33 2020

@author: yan10
"""


from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from FeatureExtraction import get_dataset
from ptdec.dec import WDEC
from ptdec.model import train




# get dataset and dataloader
data_path    = '../../data/region_proposals'
batch_size   = 256
label        = 'bike'
ds_train     = get_dataset(data_path, label)
ds_train.output = 1

# pretrain 
pretrain_epochs   = 300
finetune_epochs   = 500
training_callback = None
cuda         = torch.cuda.is_available()
ds_val       = None
embedded_dim = 1000
autoencoder  = StackedDenoisingAutoEncoder(
        [embedded_dim, 500, 500, 2000, 10],
        final_activation=None
    )
if cuda:
    autoencoder.cuda()

print('Pretraining stage.')
ae.pretrain(
    ds_train,
    autoencoder,
    cuda       = cuda,
    validation = ds_val,
    epochs     = pretrain_epochs,
    batch_size = batch_size,
    optimizer  = lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
    scheduler  = lambda x: StepLR(x, 100, gamma=0.1),
    corruption = 0.2
)

print('Training stage.')
ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
ae.train(
    ds_train,
    autoencoder,
    cuda=cuda,
    validation=ds_val,
    epochs=finetune_epochs,
    batch_size=batch_size,
    optimizer=ae_optimizer,
    scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
    corruption=0.2,
    update_callback=training_callback
)


# Train a weighted DEC
# We further re-initialize our DEC by weighted
# K-means every I epochs
I = 6
# with the new weights set by Sk normalized by the
# number of positive samples in the cluster,
# defined by DSD 
# We set the positive ratio threshold as Pk ≥ 0.6
P_k = 0.6
# Initialize cluster centers using uniform K-Means
# For clustering, we use K = 50
K = 50
# We train our model for 35 epochs for all objects
MAX_EPOCHS = 35

print('WDEC stage.')
wdec = WDEC(
    cluster_number   = K,
    hidden_dimension = 10, ### TODO: what is the WDEC architecture they used (question no.5 in notebook)
    encoder          = autoencoder.encoder
)
detector = nn.sequential(
    [nn.linear(embedded_dim, 1024), 
     nn.ReLU(),
     nn.linear(1024, 1024),
     nn.ReLU(),
     nn.linear(1024, 2),
     nn.Softmax(),]
)

if cuda:
    wdec.cuda()
    detector.cuda()
    
dec_optimizer = SGD(wdec.parameters(), lr=0.01, momentum=0.9)

'''
cross-entropy loss function
'''
#region classifier we use 3 FC layers (1024,1024,2) with a ReLU
#activation in layers 1-2 and a softmax activation for the output layer
#Dropout is used for the two hidden layers with probability of 0.8
detector = nn.sequential(
    [nn.linear(embedded_dim, 1024), 
     nn.ReLU(),
     nn.Dropout(0.8),
     nn.linear(1024, 1024),
     nn.ReLU(),
     nn.Dropout(0.8),
     nn.linear(1024, 2),
     nn.Softmax(),]
)

# ADAM for optimization with a learning rate of 10−4
det_optimizer = Adam(detector.parameters(), lr=1e-4)
# The learning rate is decreased by a factor of 0.6 every 6 epochs
scheduler = StepLR(det_optimizer, step_size = 6, gamma = 0.6)

for epoch in range(MAX_EPOCHS):
    reinitKMeans = False
    if epoch%I == 0:
        reinitKMeans = True
    train(
        dataset        = ds_train,
        model          = wdec,
        epochs         = 1,
        reinitKMeans   = reinitKMeans,
        batch_size     = 256,
        optimizer      = dec_optimizer,
        stopping_delta = None, # 0.000001,
        cuda           = cuda,
    )
    
    
    ## Train a region classifier with sampled positive and negative regions 
    
    
    
    
    
















