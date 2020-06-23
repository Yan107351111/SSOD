#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:38:10 2020

@author: yanivzis@bm
"""
from DSD import get_iou
from model import SSDetector
import os
import pickle
import pretrainedmodels
from SelectiveSearch import selective_search
import sys
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class TransDataset(Dataset):
    def __init__(self, tensors, transforms = None):
        super().__init__()
        self.tensors = tensors
        self.transforms = transforms
    def __len__(self,):
        return len(self.tensors)
    def __getitem__(self, index):
        if self.transforms is not None:
            x = self.transforms(self.tensors[index].float())
            return (x,)
        return (self.tensors[index],)
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from DSD import get_iou

if torch.cuda.device_count()>1:
    torch.cuda.set_device(1)

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


def detect(detector, image_path, device = 'cpu'):
    # print('\nproposing regions\n')
    # print(f'\n@ {time.time() - start_time}\n')
    regions, bounding_boxes = selective_search(image_path, None, None, to_file = False, silent = True)
    transform = T.Compose(
            [lambda x: x.permute(2,0,1),
             T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
             ])
    ds = TransDataset(regions, transforms = transform)
    dl = DataLoader(ds, batch_size = 512)
    features = []
    # print('\nextraction region features\n')
    # print(f'\n@ {time.time() - start_time}\n')
    for regions, in dl:
        with torch.no_grad():
            # print(regions)
            #print('\nrunning through feature_extractor\n')
            #print(f'\n@ {time.time() - start_time}\n')
            regions = regions.to(device)
            features.append(feature_extractor(regions).reshape(-1,1536))
    try: features = torch.cat(features)
    except: 
        print(features[0].shape, features[-1].shape)
        features = torch.cat(features)
    ds = TensorDataset(features)
    dl = DataLoader(ds, batch_size = 512)
    predictions = []
    # print('\nclassifing regions\n')
    # print(f'\n@ {time.time() - start_time}\n')
    for feature, in dl:
        with torch.no_grad():
            feature = feature.to(device)
            predictions.append(detector(feature))
    predictions = torch.cat(predictions)
    prediction  = torch.argmax(predictions[:,1])
    return bounding_boxes[prediction]


def evaluate(model, data_path, ground_truth_path, threshold = 0.3, device = 'cpu', time_dict = None, SMT = 1):
    model._activate = False
    images = [i for i in os.listdir(data_path)
              if i.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp',))]
    IOUs = []
    # print('\nunpacking ground truth dictionary\n')
    # print(f'\n@ {time.time() - start_time}\n')
    bb_dict = pickle.load(open(ground_truth_path, 'rb'))
    for image in tqdm(images, desc = 'processing images'):
        ious = []
        if image in list(bb_dict):
            if bb_dict[image] is not None:
                ground_truths = bb_dict[image].to(device).float()
            else: continue
        else: continue ############## TODO: figure out what to do here
        image_path = os.path.join(data_path, image)
        # print('\nperforming detection\n')
        # print(f'\n@ {time.time() - start_time}\n')
        bounding_box = detect(model, image_path, device = device).reshape(1, -1).to(device).float()
        # print('\ncomputing IOU\n')
        # print(f'\n@ {time.time() - start_time}\n')
        # print(bounding_box)
        # print(ground_truths)       
        for gt_ in ground_truths:
            gt = torch.tensor([gt_[0], gt_[1], gt_[2]-gt_[0], gt_[3]-gt_[1]]).cuda()
            ious.append(get_iou(bounding_box, gt.reshape(1,-1)))
        iou = max(ious)
        IOUs.append(iou>threshold) 
    
    return torch.mean(torch.stack(IOUs).float())


if __name__ =='__main__':
    detector_path = sys.argv[1]
    data_path = sys.argv[2]
    ground_truth_path = sys.argv[3]
    device = 'cuda'
    
    feature_extractor.to(device)
    
    # print('\nunpacking model\n')
    # detector = pickle.load(open(detector_path, 'rb')).to(device)
    detector = torch.load(detector_path)

    start_time = time.time()
    temperaturs = [1]
    results = dict()
    for temp in temperaturs:
        results[temp] = evaluate(
            detector, data_path, ground_truth_path,
            device = device, SMT = temp) 
    
    print(results)
    
    
    

        
    






