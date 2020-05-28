# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:33:57 2020

@author: yan10
"""
from typing import NamedTuple, List
import io
from matplotlib import pyplot as plt
import os
import pretrainedmodels
import sys
import time
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset


class FrameRegionProposalsDataset(Dataset):
    """Region proposals from video frames dataset."""

    def __init__(self, root_dir, label, transform=None, output = 4, random_seed = 0):
        '''
        TODO:

        Parameters
        ----------
        root_dir : string
            Directory with all the image subdirectories.
        label : string
            directory holding the positive region proposals.
        transform : callable, optional
            transform to be applied on a sample to get it to the
            embedded space. The default is None.
        random_seed : int, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        '''
        self.output = output
        torch.manual_seed(random_seed)
        self.video_ref   = {}
        self.video_deref = {}
        self.all_items   = []
        video_hash = 0
        assert label in os.listdir(root_dir), f'folder {label} not found in the root directory'
        
        # creating positive item list
        for i in os.listdir(os.path.join(root_dir, label)):
            self.all_items.append(os.path.join(label, i))
            video_name = i.split(';')[1]
            if video_name not in list(self.video_ref):
                self.video_ref[video_name] = video_hash
                self.video_deref[video_hash] = video_name
                video_hash+=1
                
        # addign negative items to the list
        other_labels = [olabel for olabel in os.listdir(root_dir)
                        if olabel is not label]
        neg_labels   = torch.randint(len(other_labels), (len(self.all_items),))
        for neg_label in neg_labels:
            other_label     = other_labels[neg_label]
            regions         = os.listdir(os.path.join(root_dir,other_label))
            neg_region_ind  = torch.randint(len(regions), (1,))
            neg_region_name = os.listdir(
                os.path.join(root_dir, other_label))[neg_region_ind]
            video_name = neg_region_name.split(';')[1]
            neg_region_name = os.path.join(
                    other_labels[neg_label],
                    neg_region_name)
            while neg_region_name in self.all_items:
                neg_label       = torch.randint(len(other_labels), (1,))
                other_label     = other_labels[neg_label]
                regions         = os.listdir(
                    os.path.join(root_dir,other_label))
                neg_region_ind  = torch.randint(len(regions), (1,))
                neg_region_name = os.listdir(
                    os.path.join(root_dir, other_label))[neg_region_ind]
                video_name = neg_region_name.split(';')[1]
                neg_region_name = os.path.join(
                    other_labels[neg_label],
                    neg_region_name)
            self.all_items.append(neg_region_name)
            if video_name not in list(self.video_ref):
                self.video_ref[video_name] = video_hash
                self.video_deref[video_hash] = video_name
                video_hash+=1
        
        self.root_dir  = root_dir
        self.transform = transform
        self.label     = label

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # print(f'getting item {idx} out of {len(self)}')
        
        
        item = self.all_items[idx]
        # print(f'item = {item}')
        img_name = os.path.join(self.root_dir,item)
        image    = plt.imread(img_name)
        # image    = image.reshape(1,*image.shape)
        label    = torch.tensor(1.) if os.path.split(item)[0]==self.label else torch.tensor(0.)
        video    = torch.tensor(self.video_ref[os.path.split(item)[1].split(';')[1]])
        box      = torch.tensor([int(i) for i in item.split(';')[3:7]])
        frame    = torch.tensor(int(item.split(';')[2]))
        if self.transform:
            with torch.no_grad():
                features = self.transform(image)
                # features.requires_grad = False
        if self.output == 6:
            return features, label, idx, box, video, frame
        if self.output == 3:
            return features, label, idx
        return features
        
        
class Label(NamedTuple):
    '''
    class holding all training assisting data.
    '''
    label: int
    box: List[int]
    video: int


def get_dataloader(data_path, batch_size, label):
    '''
    TODO:

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.

    Returns
    -------
    train_dataloader : TYPE
        DESCRIPTION.

    '''
    
    model_name = 'inceptionresnetv2'
    model = pretrainedmodels.__dict__[model_name](
        num_classes=1000, pretrained='imagenet')
    model.eval()
    
    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
         to4D,
         model])
    train_dataset = FrameRegionProposalsDataset(
        root_dir  = data_path,
        label     = label,
        transform = transform,
        
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2)
    
    return train_dataloader

def to4D(tensor):
    # print(tensor.shape)
    if len(tensor.shape)>3:
        return tensor
    if len(tensor.shape)==3:
        return tensor.unsqueeze(0)
    if len(tensor.shape)==2:
        return tensor.unsqueeze(0).unsqueeze(0)


def get_dataset(data_path, label,):
    '''
    TODO:

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.

    Returns
    -------
    train_dataset : TYPE
        DESCRIPTION.

    '''
    model_name = 'inceptionresnetv2'
    model = pretrainedmodels.__dict__[model_name](
        num_classes=1000, pretrained='imagenet')
    model.eval()
    
    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
         to4D,
         model,])
    
    train_dataset = FrameRegionProposalsDataset(
        root_dir  = data_path,
        label     = label,
        transform = transform,
    )
    return train_dataset




if __name__ == '__main__':
    data_path = sys.argv[1] #'..\data\region_proposals'
    pass    
    
    
    # inputs, labels = next(iter(train_dataloader))
    # for label in labels:
    #     for ll, label in enumerate(train_dataset.classes):
    #         if ll == train_dataset.class_to_idx: print(label)
    # print(train_dataset.class_to_idx)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    