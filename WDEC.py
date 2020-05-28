# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:35:33 2020

@author: yan10
"""

from detector_train import Trainer
from DSD import DSD
from FeatureExtraction import get_dataset
from ptdec.dec import WDEC
import ptsdae.model as ae
from ptsdae.sdae import StackedDenoisingAutoEncoder
import sys
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from WDEC_train import train, DataSetExtract, PotentialScores





# get dataset and dataloader
print('preparing prerequisites')
data_path    = sys.argv[1] # '../../data/region_proposals'
batch_size   = 256
batch_num    = 100
label        = sys.argv[2] # 'bike'
print('getting dataset')
ds_train     = get_dataset(data_path, label)
print('got dataset')
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
    encoder          = autoencoder.encoder,
    positive_ratio_threshold = P_k,
)
detector = nn.Sequential(
    [nn.Linear(embedded_dim, 1024), 
     nn.ReLU(),
     nn.Linear(1024, 1024),
     nn.ReLU(),
     nn.Linear(1024, 2),
     nn.Softmax(),]
)

if cuda:
    wdec.cuda()
    detector.cuda()
    
dec_optimizer = SGD(wdec.parameters(), lr=0.01, momentum=0.9)

'''

'''
# region classifier we use 3 FC layers (1024,1024,2) with a ReLU
# activation in layers 1-2 and a softmax activation for the output layer
# Dropout is used for the two hidden layers with probability of 0.8
# cross-entropy loss function
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
det_trainer = Trainer(
    model = detector,
    loss_fn = nn.CrossEntropyLoss(),
    optimizer = det_optimizer,
    scheduler = scheduler,
    device = 'cuda' if cuda else 'cpu',
)

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
    
    # Train a region classifier with sampled positive and negative regions 
    # get all data needed to compute the potential scores.
    features, actual, idxs, boxs, videos, frames = DataSetExtract(ds_train, wdec)
    feature_list  = []
    video_list    = []
    label_list    = []
    K             = wdec.assignment.cluster_number
    idxs, indices = torch.sort(idxs)
    features      = features[indices]
    actual        = actual[indices]
    boxs          = boxs[indices]
    videos        = videos[indices]
    frames        = frames[indices]
    DCD_count     = torch.zeros((K,))
    DCD_idxs      = []
    for C in range(K):
        C_bool = wdec.assignment.cluster_predicted[:,1]==C
        C_inds = wdec.assignment.cluster_predicted[:,0][C_bool]
        feature_list.append(features[C_inds])
        video_list.append(videos[C_inds])
        label_list.append(actual[C_inds])
        video_frames = videos[C_inds]*10000 + frames[C_inds]
        ## Run DSD
        dsd = DSD(boxs[C_inds], video_frames)
        DCD_idxs.append(idxs[C_inds][dsd])
        DCD_count[C] = len(dsd)
        
    positive_idxs = torch.cat(DCD_idxs).int()
    ## Compute the potential score Sk in (1) for each cluster
    ## set τ = 50
    potential_scores = PotentialScores(
        feature_list, video_list, label_list,
    ) 
    potential_scores /= DCD_count
    
    pred_idxs = wdec.assignment.cluster_predicted[:,0].clone()
    _, pred_idxs = pred_idxs.sort()
    pred_DCD_idxs = pred_idxs[DCD_idxs]
    dcd_sample_scores = potential_scores[wdec.assignment.cluster_predicted[pred_DCD_idxs,1]]
    
    # samples from DSD labeled as positive detections
    # all other samples labeled as negatives
    labels = torch.zeros((len(ds_train),))
    labels[positive_idxs] = 1
    # set the sample distribution to uniform over the negative samples 
    # and weighed by the normalized potential scores over the DSD samples.
    sample_distribution = torch.zeros((len(ds_train),))
    sample_distribution[positive_idxs] = dcd_sample_scores/2
    sample_distribution[sample_distribution==0] = 1/(len(ds_train)-len(positive_idxs))/2
    
    sampler_det = WeightedRandomSampler(
        weights     = sample_distribution,
        num_samples = batch_num*batch_size,
        replacement = True,
    )
    ds_det = TensorDataset(features, labels)
    dl_det = DataLoader(
        dataset    = ds_det,
        batch_size = batch_size,
        sampler    = sampler_det,
    )
    det_trainer.fit(dl_det, num_epochs = 1)
    
    
    
    
    
















