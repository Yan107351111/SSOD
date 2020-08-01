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

# get dataset and dataloader
print('preparing prerequisites')
data_path    = sys.argv[1] # '../../data/region_proposals'
batch_size   = 400
batch_num    = 1000000
label        = sys.argv[2] # 'bike'
if len(sys.argv)>3:
    dataset_path = sys.argv[3]
else:
    dataset_path = 'ds_train.p'
ds_tofile = False
if len(sys.argv)>4:
    dataset_split = sys.argv[4]
else:
    dataset_split = 10
autoencoder_path = 'autoencoder.p'
wdec_path        = 'wdec.p'  
detector_path    = 'detector.p' 
print('getting dataset')
try: 
    if not ds_tofile:
        raise
    if dataset_split>1:
        ds_train = pickle.load(open(dataset_path, 'rb'))
        ds_train.restore()
    else:
        ds_train = pickle.load(open(dataset_path, 'rb')) 
        print('prepackaged dataset found and loaded')
except:
    print('no prepackaged dataset found\ncreating dataset from data_path')
    ds_train = get_dataset(data_path, label)
    print(f'size of ds: {sys.getsizeof(ds_train.tensors)}')
    print(f'nume of ds: {torch.numel(ds_train.tensors)}')
    print(f'tots of ds: {float(torch.numel(ds_train.tensors))*sys.getsizeof(ds_train.tensors)/2**30}')
    if ds_tofile:
        if dataset_split>1:
            ds_train.to_pickle(dataset_path, dataset_split)
            
        #    ds_train_frags = ds_train.split(dataset_split)
        #    for i, ds_train_frag in enumerate(ds_train_frags):
        #        frag_name = dataset_path[:-2]+str(i)+dataset_path[-2:]
        #        pickle.dump(ds_train_frag, open(frag_name, 'wb'))
        else:
            pickle.dump(ds_train, open(dataset_path, 'wb'))
print('got dataset')
ds_train.output = 2

# pretrain 
pretrain_epochs   = 300
finetune_epochs   = 500
training_callback = None
cuda         = torch.cuda.is_available()
ds_val       = None
embedded_dim = get_embedded_dim()

try: autoencoder = pickle.load(open(autoencoder_path, 'rb'))
except:
    autoencoder = StackedDenoisingAutoEncoder(
            feature_extractor = feature_extractor,
            dimensions = [embedded_dim, 500, 500, 2000, 10],
            final_activation = None,
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
        cuda            = cuda,
        validation      = ds_val,
        epochs          = finetune_epochs,
        batch_size      = batch_size,
        optimizer       = ae_optimizer,
        scheduler       = StepLR(ae_optimizer, 100, gamma=0.1),
        corruption      = 0.2,
        update_callback = training_callback
    )
    pickle.dump(autoencoder, open(autoencoder_path, 'wb'))

ds_train.output = 6
# Train a weighted DEC
# We further re-initialize our DEC by weighted
# K-means every I epochs
I = 6
# with the new weights set by Sk normalized by the
# number of positive samples in the cluster,
# defined by DSD 
# We set the positive ratio threshold as Pk ? 0.6
P_k = 0.6
# Initialize cluster centers using uniform K-Means
# For clustering, we use K = 50
K = 50
# We train our model for 35 epochs for all objects
MAX_EPOCHS = 35

print('WDEC stage.')
print(f'@ {time.time() - start_time}')
wdec = WDEC(
    cluster_number    = K,
    hidden_dimension  = 10, ### TODO: what is the WDEC architecture they used (question no.5 in notebook)
    feature_extractor = feature_extractor,
    encoder           = autoencoder.encoder,
    positive_ratio_threshold = P_k,
)

# region classifier we use 3 FC layers (1024,1024,2) with a ReLU
# activation in layers 1-2 and a softmax activation for the output layer
# Dropout is used for the two hidden layers with probability of 0.8
# cross-entropy loss function
detector = SSDetector(embedded_dim)

if cuda:
    wdec.cuda()
    detector.cuda()
    
dec_optimizer = SGD(wdec.parameters(), lr=0.01, momentum=0.9)

'''

'''

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
    reinitKMeans = False
    if epoch%I == 0:
        reinitKMeans = True
        
    #print('\n\n\n')
    #print(f'epoch: {epoch} ;reinitKMeans: {reinitKMeans}')
    #print('\n\n\n')
      
    print('\nTraining WDEC')  
    print(f'@ {time.time() - start_time}\n')
    train(
        dataset        = ds_train,
        wdec           = wdec,
        epochs         = 1,
        reinitKMeans   = reinitKMeans,
        batch_size     = 256,
        optimizer      = dec_optimizer,
        stopping_delta = None, # 0.000001,
        cuda           = cuda,
        start_time     = start_time
    )
    
    #print('\n\n\n')
    print('\nPreparing detector training dataloader')
    print(f'@ {time.time() - start_time}\n')
    #print('\n\n\n')
    
    # Train a region classifier with sampled positive and negative regions 
    # get all data needed to compute the potential scores.
    features, actual, idxs, boxs, videos, frames = DataSetExtract(ds_train, cuda = False)
    
    
    
    #print('\n\n\n')
    #print(f'features = {features}')
    #print('\n\n\n')
    
    
    
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
        C_inds = wdec.assignment.cluster_predicted[:,0][C_bool].long()
        feature_list.append(features[C_inds])
        video_list.append(videos[C_inds])
        label_list.append(actual[C_inds])
        video_frames = videos[C_inds]*10000 + frames[C_inds]
        ## Run DSD
        dsd = DSD(boxs[C_inds], video_frames).long()
        DCD_idxs.append(idxs[C_inds][dsd])
        DCD_count[C] = len(dsd)
    
    positive_idxs = torch.cat(DCD_idxs).long()
    ## Compute the potential score Sk in (1) for each cluster
    ## set ? = 50
    potential_scores = PotentialScores(
        feature_list, video_list, label_list,
    ) 
    potential_scores[DCD_count>0] /= DCD_count[DCD_count>0]
    
    #print('\n\n\n')
    #print(f'potential_scores = {potential_scores}')
    #print('\n\n\n')
    
    pred_idxs = wdec.assignment.cluster_predicted[:,0].clone()
    _, pred_idxs = pred_idxs.sort()
    # print('\n\n\n')
    # print(f'pred_idxs = {pred_idxs}')
    # print('\n\n\n')
    # print(f'DCD_idxs = {DCD_idxs}')
    # print('\n\n\n')
    pred_DCD_idxs = pred_idxs[positive_idxs].long()
    dcd_sample_scores = potential_scores[wdec.assignment.cluster_predicted[pred_DCD_idxs,1].long()]
    
    # samples from DSD labeled as positive detections
    # all other samples labeled as negatives
    labels = torch.zeros((len(ds_train),),)
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
    print('\nTraining detector')
    print(f'@ {time.time() - start_time}\n')
    det_trainer.fit(dl_det, num_epochs = 1, start_epoch = epoch)
    
    pickle.dump(wdec, open(wdec_path, 'wb'))
    pickle.dump(detector, open(detector_path, 'wb'))
    
    
    
    
    
















