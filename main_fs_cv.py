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
#%%
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
#%%
# get dataset and dataloader
print('preparing prerequisites')
data_path     = '../../datasets/bike_2020_08_09' # '../../datasets/dog_2020_08_05' # '../../datasets/horse_2020_08_03' 
folds_K       = 5
batch_size    = 400
batch_num     = 1000
label         = 'bike' # 'dog' # 'horse'
bb_dict_path  = f'../../SSOD/{label}_bb_dict.p'
bb_dict       = pickle.load(open(bb_dict_path, 'rb'))
detector_path = 'detector.p' 
negative_freq = 0.75 ### workflow y202007251
positive_freq = 0.25 ### workflow y2020072510
MAX_EPOCHS    = 35
embedded_dim  = get_embedded_dim()
cuda = torch.cuda.is_available
device = 'cuda' if cuda else 'cpu'
#true_region_dest = '/tcmldrive/Yann/datasets/dataset_horse_2020_07_24_3/positive_regions'
#%%

fold_train_scores_3 = []
fold_valid_scores_3 = []
fold_train_scores_5 = []
fold_valid_scores_5 = []

for fold in range(folds_K):
    print(f'\n\n### Running fold: {fold} ###\n\n', flush=True)
    fold_path = os.path.join(data_path, f'fold_{fold}')
    print('getting dataset', flush=True)
    # time.sleep(5)
    # print('\n\n\n\n\n')
    # time.sleep(5)
    train_pathes = [os.path.join(fold_path, 'positive_train')]
    valid_pathes = [os.path.join(fold_path, 'positive_valid')]
    ds_train = get_dataset_transformed(train_pathes, label, silent = True)
    ds_valid = get_dataset_transformed(valid_pathes, label, silent = True)
    
    ds_train.add_full_supervision(bb_dict, iou_threshold = 0.5)
    ds_valid.add_full_supervision(bb_dict, iou_threshold = 0.5)
    sum_lables =  len(ds_train.true_labels) ### workflow y202007250 
    sum_posit = torch.sum(ds_train.true_labels) ### workflow y202007250 
    print(f'sum_posit = {sum_posit}', flush=True)
    print(f'sum_lables = {sum_lables}', flush=True)
    print(f'posit_ratio = {sum_posit/sum_lables}', flush=True)
    #train_item_file = open(f'train_items_f{fold}.txt','w+')
    #valid_item_file = open(f'valid_items_f{fold}.txt','w+')
    #train_item_file.writelines(ds_train.all_items)
    #valid_item_file.writelines(ds_valid.all_items)
    #train_item_file.close()
    #valid_item_file.close()
    
    
    sample_distribution = torch.zeros((len(ds_train),)) 
    sample_distribution[ds_train.true_labels == 0] = negative_freq/(sum_lables-sum_posit)
    sample_distribution[ds_train.true_labels == 1] = positive_freq/sum_posit 
    sampler = WeightedRandomSampler(
            weights     = sample_distribution,
            num_samples = batch_num*batch_size,
            replacement = True,
    ) 
    ds_train.output = 3  
    ds_valid.output = 3
    dlt = DataLoader(
            dataset    = ds_train,
            batch_size = batch_size,
            sampler    = sampler,
    ) 
    dlv = DataLoader(
            dataset    = ds_valid,
            batch_size = batch_size,
    ) 

    # region classifier we use 3 FC layers (1024,1024,2) with a ReLU
    # activation in layers 1-2 and a softmax activation for the output layer
    # Dropout is used for the two hidden layers with probability of 0.8
    # cross-entropy loss function
    detector = SSDetector(embedded_dim)
    detector.train_labels = torch.ones(sum_lables)*(-1)
    detector.test_labels  = torch.ones(sum_lables)*(-1)
    detector.to(device)
    

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
        # print('\nTraining detector')
        # print(f'@ {time.time() - start_time}\n')
        det_trainer.fit(dlt, dlv, num_epochs = 1, start_epoch = epoch, train_time_testing = 1)
        
    # torch.save(detector.cpu(), detector_path)
    # print('training stats')
    # print(f'true positives: {torch.sum(detector.train_labels[ds_train.true_labels==1] == 1)}')
    # print(f'total positives: {torch.sum(ds_train.true_labels==1)}')
    # print(f'true negatives: {torch.sum(detector.train_labels[ds_train.true_labels==0] == 0)}')
    # print(f'total negatives: {torch.sum(ds_train.true_labels==0)}')
    # print('\n')
    # print('testing stats')
    # print(f'true positives: {torch.sum(detector.test_labels[ds_train.true_labels==1] == 1)}')
    # print(f'total positives: {torch.sum(ds_train.true_labels==1)}')
    # print(f'true negatives: {torch.sum(detector.test_labels[ds_train.true_labels==0] == 0)}')
    # print(f'total negatives: {torch.sum(ds_train.true_labels==0)}')
    # pickle.dump(detector, open(detector_path, 'wb'))
    detector._activate = False
    
    image_source = os.path.join(fold_path, 'positive_train')
    thresholds = [0.3, 0.5]
    ds_train.output = 6
    ds_valid.output = 6
    detector.cuda()
    image_dict = {}
    for image_name in os.listdir(image_source):
        image_dict[image_name] = []
    for ii in tqdm(range(len(ds_train)), desc = 'processing images'):
        image, lable, idx, box, video, frame = ds_train[ii]
        positive_name = f'{label};{ds_train.video_deref[video.item()]};{frame:04}.png'
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
    IOUs3 = []
    IOUs5 = []
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
        IOUs3.append(iou>0.3)
        IOUs5.append(iou>0.5)
    print(f'final training score fold {fold} [iou > 0.3]: {torch.mean(torch.stack(IOUs3).float())}', flush=True) 
    print(f'final training score fold {fold} [iou > 0.5]: {torch.mean(torch.stack(IOUs5).float())}', flush=True)
    fold_train_scores_3.append(torch.mean(torch.stack(IOUs3).float()))
    fold_train_scores_5.append(torch.mean(torch.stack(IOUs5).float()))
        
    image_source = os.path.join(fold_path, 'positive_valid')
    ds_train.output = 6
    detector.cuda()
    image_dict = {}
    for image_name in os.listdir(image_source):
        image_dict[image_name] = []
    for ii in tqdm(range(len(ds_valid)), desc = 'processing images'):
        image, lable, idx, box, video, frame = ds_valid[ii]
        positive_name = f'{label};{ds_valid.video_deref[video.item()]};{frame:04}.png'
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
    IOUs3 = []
    IOUs5 = []
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
        IOUs3.append(iou>0.3)
        IOUs5.append(iou>0.5)
    print(f'final validation score fold {fold} [iou > 0.3]: {torch.mean(torch.stack(IOUs3).float())}', flush=True) 
    print(f'final validation score fold {fold} [iou > 0.5]: {torch.mean(torch.stack(IOUs5).float())}', flush=True)
    fold_valid_scores_3.append(torch.mean(torch.stack(IOUs3).float()).item())
    fold_valid_scores_5.append(torch.mean(torch.stack(IOUs5).float()).item())

print(f'final training scores [iou > 0.3]: {fold_train_scores_3}', flush=True)
print(f'final training scores [iou > 0.5]: {fold_train_scores_5}', flush=True)
print(f'mean training score [iou > 0.3]: {torch.mean(torch.tensor(fold_train_scores_3)).item()}', flush=True) 
print(f'mean training score [iou > 0.5]: {torch.mean(torch.tensor(fold_train_scores_5)).item()}', flush=True)
print(f'final validation scores [iou > 0.3]: {fold_valid_scores_3}', flush=True)
print(f'final validation scores [iou > 0.5]: {fold_valid_scores_5}', flush=True)  
print(f'mean validation score [iou > 0.3]: {torch.mean(torch.tensor(fold_valid_scores_3)).item()}', flush=True) 
print(f'mean validation score [iou > 0.5]: {torch.mean(torch.tensor(fold_valid_scores_5)).item()}', flush=True)







