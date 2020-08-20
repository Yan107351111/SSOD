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
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
from tqdm import tqdm
from typing import List, Optional

class TransDataset(Dataset):
    '''
    Dataset class. Same as the Tensot Dataset, but supports transformes.
    '''
    def __init__(self, tensors: List[torch.Tensor], transforms: Optional[object] = None):
        super().__init__()
        self.tensors = tensors
        self.transforms = transforms
    
    def __len__(self,):
        return len(self.tensors)
    
    def __getitem__(self, index):
        if self.transforms is not None:
            x = self.transforms(self.tensors[index]).float()
            return (x,)
        return (self.tensors[index],)

if torch.cuda.device_count()>1:
    torch.cuda.set_device(1)

# load feature extraction backbone.
model_name = 'inceptionresnetv2'
try:
    inception_resnet_v2 = pickle.load(open('../'+model_name+'.p', 'rb'))
except:
    inception_resnet_v2 = pretrainedmodels.__dict__[model_name](
        num_classes=1000, pretrained='imagenet')
inception_resnet_v2_children = [child for child in inception_resnet_v2.children()]
feature_extractor = nn.Sequential(*inception_resnet_v2_children[:-1])#[:-2], nn.Flatten())
feature_extractor.eval()
del inception_resnet_v2_children, inception_resnet_v2


def get_embedded_dim(in_shape: tuple = (3,299,299)):
    '''
    Retrive the dimention of the embedded space.

    Parameters
    ----------
    in_shape : tuple, optional
        DESCRIPTION. The default is (3,299,299).

    Returns
    -------
    int.
        The dimention of the output embedding.
    '''
    _in = torch.rand(1, *in_shape)
    _out = feature_extractor(_in)
    return _out.shape[1]

embedded_dim = get_embedded_dim()

def detect(detector: nn.Module, image_path: str, device: str = 'cpu'):
    '''
    Apply the detector on all the images in the image_path.

    Parameters
    ----------
    detector : nn.Module
        The region classifier.
    image_path : str
        Path to input images.
    device : str, optional
        Device to use for inference. The default is 'cpu'.

    Returns
    -------
    Tuple[bounding box, P(C=1|r)]
        the bounding box of the highest confidence region and the probability 
        the region (r) contains the object (C=1).
    '''
    torch.manual_seed(0)
    # print('\nproposing regions\n')
    # print(f'\n@ {time.time() - start_time}\n')
    # Retriving ROIs using selective_search
    _, regions, bounding_boxes = selective_search(image_path, None, None, to_file = False, silent = True)
    
    # forming data containers for feature extraction.
    transform = T.Compose(
            [T.ToTensor(),
             T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
             ])
    ds = TransDataset(regions, transforms = transform)
    dl = DataLoader(ds, batch_size = 512)
    
    # transforming ROIs to feature vectors.
    features = []
    # print('\nextraction region features\n')
    # print(f'\n@ {time.time() - start_time}\n')
    feature_extractor.to(device)
    for regions, in dl:
        with torch.no_grad():
            # print(regions)
            #print('\nrunning through feature_extractor\n')
            #print(f'\n@ {time.time() - start_time}\n')
            regions = regions.to(device)
            features.append(feature_extractor(regions).reshape(-1,embedded_dim))
    features = torch.cat(features)
    
    # forming data containers for classification.
    ds = TensorDataset(features)
    dl = DataLoader(ds, batch_size = 512)
    
    # classifing regions.
    predictions = []
    # print('\nclassifing regions\n')
    # print(f'\n@ {time.time() - start_time}\n')
    for feature, in dl:
        with torch.no_grad():
            feature = feature.to(device)
            predictions.append(detector(feature))
    predictions = torch.cat(predictions)
    prediction  = torch.argmax(predictions[:,1])
    return bounding_boxes[prediction], nn.functional.softmax(predictions[prediction,:])


def evaluate(model, data_path, ground_truth_path, threshold = 0.5, device = 'cpu'):
    # deactivating classifier output activation to prevent loss of confidense information.
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
        bounding_box, probability = detect(model, image_path, device = device)#, cheat = ground_truths[0])
        bounding_box = bounding_box.reshape(1, -1).to(device).float()
        # print('\ncomputing IOU\n')
        # print(f'\n@ {time.time() - start_time}\n')
        # print(bounding_box)
        # print(ground_truths)       
        for gt_ in ground_truths:
            gt = torch.tensor([gt_[0], gt_[1], gt_[2]-gt_[0], gt_[3]-gt_[1]]).cuda()
            ious.append(get_iou(bounding_box, gt.reshape(1,-1)))
        iou = max(ious)
        #print(f'probability = {probability}')
        #print(f'{image} iou: {iou}')
        IOUs.append(iou>threshold) 
    
    return torch.mean(torch.stack(IOUs).float())


if __name__ =='__main__':
    K = 5
    label = 'dog'
    results = torch.ones(K, dtype = float)*(-1.)
    for fold in range(K):
        print(f'Proccessing fold {fold}:\n')
        detector_path = f'detector_F{fold}.p' # sys.argv[1]#'../20200622_horse_fs/20200720/detector.p'  #
        data_path = f'/tcmldrive/Yann/datasets/dog_2020_08_05/fold_{fold}/positive_valid/' # f'/tcmldrive/Yann/datasets/bike_2020_08_10/fold_{fold}/positive_train/' #  # sys.argv[2]#'../datasets/dataset_horse_2020_06_08/positive/'#
        ground_truth_path  = f'../../SSOD/{label}_bb_dict.p'#'horse_bb_dict.p'#
        device = 'cuda'
        
        feature_extractor.to(device)
        
        # print('\nunpacking model\n')
        detector = pickle.load(open(detector_path, 'rb')).to(device)
        detector.eval()
        detector.train(False)
        # detector = torch.load(detector_path).to(device)
    
        start_time = time.time()
        result = evaluate(
            detector, data_path, ground_truth_path,
            device = device,) 
        print('')
        print(result)
        print('\n')
        results[fold] = result
    print(f'results: \n{results}\n')
    print(f'mean: \n{torch.mean(results)}\n')
    
    
    



