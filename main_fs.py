# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:35:33 2020

@author: yan10
"""

from detector_train import DetectorTrainer
from DSD import DSD, get_iou
from FeatureExtraction import get_dataset, get_dataset_transformed, get_embedded_dim, get_dataset_transformed_from_file
from model import SSDetector
import os
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
from tqdm import tqdm
from utils import compressed_pickle, decompress_pickle
from wdec import train, DataSetExtract, PotentialScores

start_time = time.time()
if torch.cuda.device_count()>1: torch.cuda.set_device(1)
torch.manual_seed(0)

model_name = 'inceptionresnetv2'
try:
    inception_resnet_v2 = pickle.load(open('../'+model_name+'.p', 'rb'))
except:
    inception_resnet_v2 = pretrainedmodels.__dict__[model_name](
        num_classes=1000, pretrained='imagenet')
inception_resnet_v2_children = [child for child in inception_resnet_v2.children()]
feature_extractor = nn.Sequential(*inception_resnet_v2_children[:-1])
feature_extractor.eval()
del inception_resnet_v2_children, inception_resnet_v2

# get dataset and dataloader
print('preparing prerequisites')
data_path     = sys.argv[1] # '../../datasets/dataset_horse_2020_07_24_3'
batch_size    = 400
batch_num     = 50
label         = sys.argv[2] # 'horse'
bb_dict_path  = sys.argv[3] # '../../SSOD/horse_bb_dict.p'
bb_dict       = pickle.load(open(bb_dict_path, 'rb'))
detector_path = 'detector.p' 
negative_freq = 0.5 ### workflow y202007251
positive_freq = 0.5 ### workflow y202007251

true_region_dest = '/tcmldrive/Yann/datasets/dataset_horse_2020_07_23/positive_regions'

print('getting dataset')
ds_train = get_dataset_transformed(data_path, label)
ds_files = get_dataset_transformed_from_file(os.path.join(data_path, 'regions'), label)

direct = {}
from_file = {}
residuals = torch.zeros(len(ds_train), dtype = float)
for i in range(len(ds_train)):
    direct[ds_train.all_items[i][:-8]] = ds_train.tensors[i]
for i in range(len(ds_files)):
    from_file[ds_files.all_items[i][:-8]] = ds_files.tensors[i]

for i in range(len(ds_train)):
    region_name = ds_train.all_items[i][:-8]
    if region_name in list(from_file):
        residuals[i] = torch.sum(torch.abs(direct[region_name].cpu() - from_file[region_name].cpu()))
    else:
        residuals[i] = -1 
    
print(f'residuals: {residuals}')

raise




ds_train.add_full_supervision(bb_dict, iou_threshold = 0.5)#, true_cp = True, true_cp_dest = true_region_dest)
sum_lables =  len(ds_train.true_labels) ### workflow y202007250 
sum_posit = torch.sum(ds_train.true_labels) ### workflow y202007250 
print(f'sum_posit = {sum_posit}')
print(f'sum_lables = {sum_lables}')
print(f'posit_ratio = {sum_posit/sum_lables}')


print('got dataset')

sample_distribution = torch.zeros((len(ds_train),)) ### workflow y202007251 
sample_distribution[ds_train.true_labels == 0] = negative_freq/(sum_lables-sum_posit) ### workflow y202007251
sample_distribution[ds_train.true_labels == 1] = positive_freq/sum_posit ### workflow y202007251
sampler = WeightedRandomSampler(
        weights     = sample_distribution,
        num_samples = batch_num*batch_size,
        replacement = True,
) ### workflow y202007251
ds_train.output = 3 
print(f'ds_train._transformed = {ds_train._transformed}') 
dl = DataLoader(
        dataset    = ds_train,
        batch_size = batch_size,
        sampler    = sampler,    ### workflow y202007251
) 
# pretrain 
embedded_dim = get_embedded_dim()
print(f'embedded_dim = {embedded_dim}')
MAX_EPOCHS = 35

print('WDEC stage.')
print(f'@ {time.time() - start_time}')

# region classifier we use 3 FC layers (1024,1024,2) with a ReLU
# activation in layers 1-2 and a softmax activation for the output layer
# Dropout is used for the two hidden layers with probability of 0.8
# cross-entropy loss function
detector = SSDetector(embedded_dim)
detector.train_labels = torch.ones(sum_lables)*(-1)
detector.test_labels  = torch.ones(sum_lables)*(-1)
cuda = torch.cuda.is_available
if cuda:
    detector.cuda()
device = 'cuda' if cuda else 'cpu'

# ADAM for optimization with a learning rate of 10?4
det_optimizer = Adam(detector.parameters(), lr=1e-4)
# The learning rate is decreased by a factor of 0.6 every 6 epochs
scheduler = StepLR(det_optimizer, step_size = 6, gamma = 0.6)

# nw, pw = sum_posit/sum_lables, (sum_lables-sum_posit)/sum_lables ### workflow y202007250 
CEL = nn.CrossEntropyLoss()# weight = torch.tensor([nw, pw]).to(device)) ### workflow y202007250 
def loss(y_hat, y_true):
    return CEL(y_hat, y_true.long())

det_trainer = DetectorTrainer(
    model = detector,
    loss_fn = loss,
    optimizer = det_optimizer,
    scheduler = scheduler,
    device = device,
)

for epoch in range(MAX_EPOCHS):
      
    # Train a region classifier with sampled positive and negative regions 
    # get all data needed to compute the potential scores.
    print('\nTraining detector')
    print(f'@ {time.time() - start_time}\n')
    det_trainer.fit(dl, dl, num_epochs = 1, start_epoch = epoch)
    
torch.save(detector.cpu(), detector_path)
print('training stats')
print(f'true positives: {torch.sum(detector.train_labels[ds_train.true_labels==1] == 1)}')
print(f'total positives: {torch.sum(ds_train.true_labels==1)}')
print(f'true negatives: {torch.sum(detector.train_labels[ds_train.true_labels==0] == 0)}')
print(f'total negatives: {torch.sum(ds_train.true_labels==0)}')
print('\n')
print('testing stats')
print(f'true positives: {torch.sum(detector.test_labels[ds_train.true_labels==1] == 1)}')
print(f'total positives: {torch.sum(ds_train.true_labels==1)}')
print(f'true negatives: {torch.sum(detector.test_labels[ds_train.true_labels==0] == 0)}')
print(f'total negatives: {torch.sum(ds_train.true_labels==0)}')
# pickle.dump(detector, open(detector_path, 'wb'))
detector._activate = False

image_source = '/tcmldrive/Yann/datasets/dataset_horse_2020_07_23/positive'
threshold = 0.3
ds_train.output = 6
detector.cuda()
image_dict = {}
for image_name in os.listdir(image_source):
    image_dict[image_name] = []
for ii in tqdm(range(len(ds_train)), desc = 'processing images'):
    image, label, idx, box, video, frame = ds_train[ii]
    positive_name = f'horse;{ds_train.video_deref[video.item()]};{frame:04}.png'
    if positive_name in list(bb_dict):
        if bb_dict[positive_name] is not None:
            ground_truths = bb_dict[positive_name].to(device).float()
        else: continue
    else: continue
    with torch.no_grad():
        scores = detector(image.cuda())
    score = scores[1]
    if len(image_dict[positive_name])!= 2:
        image_dict[positive_name] = [score, box]
    elif image_dict[positive_name][0] < score:
        image_dict[positive_name] = [score, box]
IOUs = []
images = [i for i in os.listdir(image_source)
          if i.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp',))]
for image in tqdm(list(image_dict)):
    if len(image_dict[image]) != 2:
        continue
    ious = []
    if image in list(bb_dict):
        if bb_dict[image] is not None:
            ground_truths = bb_dict[image].to(device).float()
        else: continue
    else: continue
    bounding_box = image_dict[image][1].reshape(1, -1).to(device).float()  
    for gt_ in ground_truths:
        gt = torch.tensor([gt_[0], gt_[1], gt_[2]-gt_[0], gt_[3]-gt_[1]]).cuda()
        ious.append(get_iou(bounding_box, gt.reshape(1,-1)))
    iou = max(ious)
    IOUs.append(iou>threshold)
print(f'final training score: {torch.mean(torch.stack(IOUs).float())}') 
    
















