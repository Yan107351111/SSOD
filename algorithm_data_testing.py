# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:35:33 2020

@author: yan10
"""
import cv2
from detector_train import DetectorTrainer
from DSD import DSD
from FeatureExtraction import get_dataset, get_embedded_dim
from matplotlib import pyplot as plt
from model import SSDetector
import os
import pickle
import pretrainedmodels
import sys
import time
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader
from tqdm import tqdm


start_time = time.time()
if torch.cuda.device_count()>1:
    torch.cuda.set_device(1)
torch.manual_seed(0)

# get dataset and dataloader
print('preparing prerequisites')
data_path     = sys.argv[1] # '../../data/region_proposals'
label         = sys.argv[2]
bb_dict_path  = sys.argv[3]
positive_path = sys.argv[4]
bb_dict       = pickle.load(open(bb_dict_path, 'rb'))

detection_dest = '/tcmldrive/Yann/datasets/dataset_horse_2020_07_23/fs_detections'

print('getting dataset')
ds_train = get_dataset(data_path, label)
ds_train.add_full_supervision(bb_dict, iou_threshold = 0.5,)
sum_lables =  len(ds_train.true_labels) ### workflow y202007250 
sum_posit = torch.sum(ds_train.true_labels) ### workflow y202007250 
print(f'sum_posit = {sum_posit}')
print(f'sum_lables = {sum_lables}')
print(f'posit_ratio = {sum_posit/sum_lables}')

ds_train.output = 6
positive_dict = {}
for positive_name in os.listdir(positive_path):
    positive_image_path = os.path.join(positive_path, positive_name)
    positive_dict[positive_name] = plt.imread(positive_image_path)
for ii in tqdm(range(len(ds_train)), desc = 'processing images'):
    if ds_train.true_labels[ii] == 1:
        image, label, idx, box, video, frame = ds_train[ii]
        positive_name = f'horse;{ds_train.video_deref[video.item()]};{frame:04}.png'
        pt1 = tuple([box[0].int().item(), box[1].int().item()])
        pt2 = tuple([box[0].int().item()+box[2].int().item(), box[1].int().item()+box[3].int().item()])
        positive_dict[positive_name] = cv2.rectangle(positive_dict[positive_name], pt1, pt2, 1, 3)
for positive_name in os.listdir(positive_path):
    positive_image_path = os.path.join(positive_path, positive_name)
    plt.imsave(detection_dest, positive_dict[positive_name])






























