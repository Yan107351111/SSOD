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

from FeatureExtraction import FrameRegionProposalsDataset, Label, get_dataloader, to4D, get_dataset

class Demo:
    class FrameRegionProposalsDataset(Dataset):
        """Region proposals from video frames dataset."""
    
        def __init__(self, root_dir, label, transform=None, random_seed = 0):
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
            print(other_labels)
            neg_labels   = torch.randint(len(other_labels), (len(self.all_items),))
            for neg_label in neg_labels:
                other_label     = other_labels[neg_label]
                regions         = os.listdir(os.path.join(root_dir,other_label))
                neg_region_ind  = torch.randint(len(regions), (1,))
                neg_region_name = os.listdir(os.path.join(root_dir, other_label))[neg_region_ind]
                video_name = neg_region_name.split(';')[1]
                neg_region_name = os.path.join(
                    other_labels[neg_label],
                    neg_region_name)
                while neg_region_name in self.all_items:
                    neg_label       = torch.randint(len(other_labels), (1,))
                    other_label     = other_labels[neg_label]
                    regions         = os.listdir(os.path.join(root_dir,other_label))
                    neg_region_ind  = torch.randint(len(regions), (1,))
                    neg_region_name = os.listdir(os.path.join(root_dir, other_label))[neg_region_ind]
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
            print('getting item')
            if torch.is_tensor(idx):
                idx = idx.tolist()
            
            img_name = os.path.join(self.root_dir,self.all_items[idx])
            print(self.root_dir)
            print(self.all_items[idx])
            print(img_name)
            image    = plt.imread(img_name)
            # image    = image.reshape(1,*image.shape)
            label    = torch.tensor(1.) if self.all_items[idx].split('\\')[0]==self.label else torch.tensor(0.)
            video    = torch.tensor(self.video_ref[self.all_items[idx].split('\\')[1].split(';')[1]])
            box      = torch.tensor([int(i) for i in self.all_items[idx].split(';')[3:7]])
            if self.transform:
                with torch.no_grad():
                    features = self.transform(image)
                    # features.requires_grad = False
            print(f'features.requires_grad = {features.requires_grad}')
            print(f'label.requires_grad = {label.requires_grad}')
            return features, label , box, video
    
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
        print('getting dataset')
        train_dataset = FrameRegionProposalsDataset(
            root_dir  = data_path,
            label     = label,
            transform = transform,
            
        )
        print('inserting dataset into dataloader')
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
    
    def get_dataset(data_path, batch_size,):
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
                root      = data_path,
                transform = transform,
            )
            return train_dataset
        


if __name__ == '__main__':
    start = time.time()
    data_path = '../region_proposals_example' #'../../data/region_proposals'
    print('getting data loader')
    train_dataloader = get_dataloader(
        data_path = data_path, batch_size = 5, label = 'dog'
    )
    print('getting samples')
    print(time.time()-start)
    with torch.no_grad():
        features, labels, boxs, videos = next(iter(train_dataloader))
    # for label in labels:
    #     for ll, label in enumerate(train_dataset.classes):
    #         if ll == train_dataset.class_to_idx: print(label)
    print('printing samples')
    print(f'labels = {labels}')
    print(f'boxs = {boxs}')
    print(f'videos = {videos}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    