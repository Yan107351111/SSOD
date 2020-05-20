# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:33:57 2020

@author: yan10
"""
import io
import os
import pretrainedmodels
import sys
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import dataset


class FrameRegionProposalsDataset(dataset):
    """Region proposals from video frames dataset."""

    def __init__(self, root_dir, label, transform=None, random_seed = 0):
        """
        Args:
            root_dir (string): Directory with all the image subdirectories.
            label    (string): The name of the class. also, the name of the
                directory holding the positive region proposals.
            transform (callable, optional): transform to be applied on a 
                sample to get it to the embedded space.
        """
        torch.manual_seed(random_seed)
        self.video_ref   = {}
        self.video_deref = {}
        self.all_items   = []
        frame_hash = 0
        assert label in os.listdir(root_dir), f'folder {label} not found in the root directory'
        
        # creating positive item list
        for i in os.listdir(os.path.join(root_dir, label)):
            self.all_items.append(os.path.join(label, i))
            frame_name = i.split(';')[1]
            if frame_name not in list(self.frame_ref):
                self.video_ref[frame_name] = frame_hash
                self.video_deref[frame_hash] = frame_name
                frame_hash+=1
                
        # addign negative items to the list
        other_labels = [olabel for olabel in os.path.join(root_dir)
                        if olabel is not label]
        neg_labels   = torch.randint(len(other_labels), len(self.all_items))
        for neg_label in neg_labels:
            other_label     = other_labels[neg_label]
            regions         = os.listdir(other_label)
            neg_region_ind  = torch.rand(len(regions), (1,))
            neg_region_name = os.path.join(
                other_labels[neg_label],
                os.listdir(os.path.join(root_dir, neg_label))[neg_region_ind])
            while neg_region_name in self.all_items:
                neg_label = torch.randint(len(other_labels), (1,))
                other_label     = other_labels[neg_label]
                neg_region_ind  = torch.rand(len(regions), (1,))
                neg_region_name = os.path.join(
                    other_labels[neg_label],
                    os.listdir(os.path.join(root_dir, neg_label))[neg_region_ind])
            self.all_items.append((os.path.join(neg_label, neg_region_name)))
        
        self.root_dir  = root_dir
        self.transform = transform
        self.label     = label

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,self.all_items[idx])
        image    = io.imread(img_name)
        label    = torch.tensor(1.) if self.all_items[idx].split('\\')[0]==self.label else torch.tensor(0.)
        video    = torch.tensor(self.video_ref[self.all_items[idx].split('\\')[1].split(';')[1]])
        box      = torch.tensor([int(i) for i in self.all_items[idx].split(';')[3:7]])
        if self.transform:
            features = self.transform(image)
        return features, label, box, video


def get_dataloader(data_path, batch_size,):
    
    model_name = 'inceptionresnetv2'
    model = pretrainedmodels.__dict__[model_name](
        num_classes=1000, pretrained='imagenet')
    model.eval()
    
    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         model])
    
    train_dataset = FrameRegionProposalsDataset(
        root      = data_path,
        transform = transform,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2)
    
    return train_dataloader

def get_dataset(data_path, batch_size,):
    
    model_name = 'inceptionresnetv2'
    model = pretrainedmodels.__dict__[model_name](
        num_classes=1000, pretrained='imagenet')
    model.eval()
    
    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         model])
    
    train_dataset = FrameRegionProposalsDataset(
        root      = data_path,
        transform = transform,
    )
    
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size,
    #     shuffle=True, num_workers=2)
    # xs = []
    # ys = []
    # for x, y in train_dataloader:
    #     xs.append(model(x))
    #     ys.append(torch.tensor(y))
    
    # train_embedded_dataset = torch.utils.data.TensorDataset(
    #     torch.cat(x), torch.cat(y)
    # )
    return train_dataset



if __name__ == '__main__':
    data_path = sys.argv[1] #'..\data\region_proposals'
    pass    
    
    
    # inputs, labels = next(iter(train_dataloader))
    # for label in labels:
    #     for ll, label in enumerate(train_dataset.classes):
    #         if ll == train_dataset.class_to_idx: print(label)
    # print(train_dataset.class_to_idx)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    