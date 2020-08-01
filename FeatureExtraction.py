# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:33:57 2020

@author: yan10
"""

from DSD import get_iou
import io
from matplotlib import pyplot as plt
import os
import pickle
import pretrainedmodels
from SelectiveSearch import selective_search
import shutil
import sys
import time
import torch
from torch import nn 
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from tqdm import tqdm
from typing import NamedTuple, List



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

cuda = torch.cuda.is_available
device = 'cuda' if cuda else 'cpu'

def get_embedded_dim(in_shape: tuple = (3,299,299)):
    _in = torch.rand(1, *in_shape)
    _out = feature_extractor(_in)
    return _out.shape[1]
    
class FrameRegionProposalsDataset(Dataset):
    """Region proposals from video frames dataset."""

    def __init__(self, 
            root_dir, label, 
            transform = None, 
            output = 6, random_seed = 0, _construct = True,
            ):
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
        if not _construct:
            return
        torch.manual_seed(random_seed)   
        self._fully_supervised = False
        self._transformed = False
        self.tensors      = None
        self.root_dir     = root_dir
        self.transform    = transform
        self.label        = label
        self.output       = output
        self.all_items    = os.listdir(root_dir)        
        self._create_mapings()
        
    def _create_mapings(self,):
        self.video_ref    = {}
        self.video_deref  = {}    
        video_hash = 0
        for i in self.all_items:
            video_name = i.split(';')[1]
            if video_name not in list(self.video_ref):
                self.video_ref[video_name]   = video_hash
                self.video_deref[video_hash] = video_name
                video_hash += 1    
        
    @classmethod
    def from_ss(cls, label, all_items, features,):
        dupe = cls(None, None, _construct = False) 
        dupe._fully_supervised = False
        dupe._transformed = True
        dupe.root_dir     = None
        dupe.transform    = None
        dupe.label        = label
        dupe.output       = 6
        dupe.all_items    = all_items
        dupe.tensors      = features
        dupe._create_mapings()
        return dupe

    def __len__(self):
        if self._transformed: return len(self.tensors)
        else: return len(self.all_items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #if self._fully_supervised:
        #    image = self.tensors[idx]
        #    label = self.true_labels[idx]
        #    return image, label 
        
        item = self.all_items[idx]
        if not self._transformed:
            img_name = os.path.join(self.root_dir,item)
            image    = plt.imread(img_name)
            if self.transform is not None:
                image    = self.transform(image)
        else: image = self.tensors[idx]
        # image    = image.reshape(1,*image.shape)
        label    = torch.tensor(1.) if item.split(';')[0]==self.label else torch.tensor(0.)
        video    = torch.tensor(self.video_ref[os.path.split(item)[1].split(';')[1]])
        box      = torch.tensor([int(i) for i in item.split(';')[3:7]])
        frame    = torch.tensor(int(item.split(';')[2]))
        
        if self._fully_supervised:
            label = self.true_labels[idx]
        idx = torch.tensor(idx)
        # if self.transform:
        #     with torch.no_grad():
        #         features = self.transform(image)
                # features.requires_grad = False
        if self.output == 6:
            return image, label, idx, box, video, frame
        if self.output == 3:
            return image, label, idx
        if self.output == 2:
            return image, label
        return image
        
    def split(self, frag_num: int = 2):
        assert frag_num>=2
        splits = [0]
        for i in range(1,frag_num):
            splits.append(i*(len(self)//frag_num))
        splits.append(len(self))
        print(splits)
        frags = []
        for i in range(frag_num):
            frag = FrameRegionProposalsDataset(None, None, _construct = False)
            frag._transformed = True
            frag.root_dir  = self.root_dir
            frag.transform = self.transform
            frag.label     = self.label
            frag.output    = self.output
            frag.video_ref   = self.video_ref
            frag.video_deref = self.video_deref
            frag.all_items   = self.all_items[splits[i]:splits[i+1]]
            if self._transformed:
                frag.tensors = self.tensors[splits[i]:splits[i+1]]
            frags.append(frag)
        return frags
        
    def append(self, other):
        if self.root_dir     != other.root_dir  \
           or self.transform != other.transform \
           or self.label     != other.label:
            raise RuntimeError('Datasets incompatible')
        self.video_ref.update(other.video_ref)
        self.video_deref.update(other.video_deref)
        self.all_items = self.all_items + other.all_items
        if self._transformed:
            self.tensors = torch.cat(self.tensors, other.tensors)
            
    def to_pickle(self,fname, frag_num = 2):
        splits = [0]
        for i in range(1,frag_num):
            splits.append(i*(len(self)//frag_num))
        splits.append(len(self))
        for i in range(frag_num):
             tensors = self.tensors[splits[i]:splits[i+1]]
             pickle.dump(tensors, open(f'tensors{i}.p', 'wb'))
             
        pickle.dump(self.duplicate(), open(fname, 'wb'))
        
    def duplicate(self):
        dupe = FrameRegionProposalsDataset(None, None, _construct = False) 
        dupe._transformed = self._transformed
        dupe.root_dir     = self.root_dir
        dupe.transform    = self.transform
        dupe.label        = self.label
        dupe.output       = self.output
        dupe.video_ref    = self.video_ref
        dupe.video_deref  = self.video_deref
        dupe.all_items    = self.all_items
        dupe.tensors      = None
        return dupe
        
    def restore(self, data_path = '.'):
        frags = [i for i in os.listdir(data_path) if i.startswith('tensor')]
        frags.sort()
        tensors = []
        for frag_name in frags:
            tensors.append(pickle.load(open(frag_name, 'rb')))
        self.tensors = torch.cat(tensors)
        
    def add_full_supervision(self, bb_dict: dict, iou_threshold: float = 0.5, true_cp: bool = False, true_cp_dest: str = '.', selection_method: str = 'threshold', K: int = 5, ):
        '''
        

        Parameters
        ----------
        bb_dict : dict
            dictionary mapping whole image name to the bounding boxs in the
            image.self
        iou_threshold : float, optional
            The threshold overwhich the region will be considered a positive
            detection.
            The default is 0.5.

        Returns
        -------
        None.

        '''
        true_regions = []
        self.true_labels = torch.zeros((len(self.all_items)))
                
        for ii, item in enumerate(self.all_items):
            lebel = item.split(';')[0]
            video = os.path.split(item)[1].split(';')[1]
            frame = item.split(';')[2]
            box   = torch.tensor([int(i) for i in item.split(';')[3:7]])
            im_name = lebel+';'+video+';'+frame+'.png'
            if im_name in list(bb_dict):
                if bb_dict[im_name] is None:
                    continue
                for gt_ in bb_dict[im_name]:
                    gt = torch.tensor([gt_[0], gt_[1], gt_[2]-gt_[0], gt_[3]-gt_[1]]).cuda()
                    if get_iou(box.reshape(1,-1).cpu().float(), gt.reshape(1,-1).cpu().float()) > iou_threshold:
                        self.true_labels[ii] = 1
                        true_regions.append(item)
        self._fully_supervised = True
        self.output = 2
        if true_cp:
            for item in true_regions:
                shutil.copyfile(os.path.join(self.root_dir, item), os.path.join(true_cp_dest, item))
            

        
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
    
    
    
    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
         ])
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
'''  
class extract_features():
    def __init__(self,):
        model_name = 'inceptionresnetv2'
        model = pretrainedmodels.__dict__[model_name](
        num_classes=1000, pretrained='imagenet')
        self.model = model.eval()
    
    def __call__(self, tensor):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            tensor = tensor.cuda()
        else:
            self.model = self.model.cpu()
            tensor = tensor.cpu()
        with torch.no_grad():
            return self.model()
'''

def get_dataset(data_path, label, sample = -1):
    transform = T.Compose(
            [T.ToTensor(),
             T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
             ])
    train_dataset = FrameRegionProposalsDataset(
        root_dir  = data_path,
        label     = label,
        transform = transform,
    )
    train_dataset.output = 2
    return train_dataset

def get_dataset_transformed_from_file(data_path, label, sample = -1):
    cuda = torch.cuda.is_available()
    transform = T.Compose(
            [T.ToTensor(),
             T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
             ])
    train_dataset = FrameRegionProposalsDataset(
        root_dir  = data_path,
        label     = label,
        transform = transform,
    )
    train_dataset.output = 1
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500,
        shuffle=True, num_workers=4)
    tensors = []z
    if cuda:
        feature_extractor.cuda()
    for ii, batch in enumerate(tqdm(train_dataloader, desc = 'extracting features:')):
        if cuda:
            batch = batch.cuda()
        with torch.no_grad():
             tensors.append(feature_extractor(batch).cpu().squeeze())
        if ii == sample-1:
            break
    tensors = torch.cat(tensors)
    # print(tensors.shape)
    train_dataset.tensors = tensors
    train_dataset._transformed = True 
    train_dataset.output = 6
    
    feature_extractor.cpu()
    return train_dataset

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
        else:
            return (self.tensors[index],)

def get_dataset_transformed(
        data_path, label, 
        embedded_dim = get_embedded_dim(), sample = -1):
    positives = ['positive'] + os.listdir(os.path.join(data_path, 'positive'))
    negatives = ['negative'] + os.listdir(os.path.join(data_path, 'negative')) 
    cuda = torch.cuda.is_available()
    transform = T.Compose(
            [lambda x: x.permute(2,0,1),
             T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
             ])
    
    feature_extractor.to(device)
    all_items = []
    all_bounding_boxes = []
    all_features = []
    for sentiment in [positives, negatives]:
        for image in sentiment[1:]:
            image_path = os.path.join(data_path, sentiment[0], image)
            names, regions, _ = selective_search(
                image_path, None, None,
                to_file = False, silent = True
            )
            #for i in range(100, 110):
            #    plt.figure()
            #    plt.imshow(regions[i])
            #    plt.title(names[i])
            #    plt.show()
            #raise
            all_items += names
            #all_bounding_boxes.append(bounding_boxes)
            ds = TransDataset(regions, transforms = transform)
            dl = DataLoader(ds, batch_size = 512)
            
            
            for regions, in dl:
                with torch.no_grad():
                    regions = regions.to(device)
                    all_features.append(
                        feature_extractor(regions).reshape(-1,embedded_dim)
                    )
    all_items = [os.path.split(item)[1] for item in all_items]
    all_features = torch.cat(all_features)
    
    #all_bounding_boxes = torch.cat(all_bounding_boxes)
    
    dataset = FrameRegionProposalsDataset.from_ss(label, all_items, all_features,)
    
            
    '''        
    train_dataset = FrameRegionProposalsDataset(
        root_dir  = data_path,
        label     = label,
        transform = transform,
    )
    train_dataset.output = 1
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500,
        shuffle=True, num_workers=4)
    tensors = []
    if cuda:
        feature_extractor.cuda()
    for ii, batch in enumerate(tqdm(train_dataloader, desc = 'extracting features:')):
        if cuda:
            batch = batch.cuda()
        with torch.no_grad():
             tensors.append(feature_extractor(batch).cpu().squeeze())
        if ii == sample-1:
            break
    tensors = torch.cat(tensors)
    # print(tensors.shape)
    train_dataset.tensors = tensors
    train_dataset._transformed = True 
    train_dataset.output = 6
    '''
    feature_extractor.cpu()
    return dataset


if __name__ == '__main__':
    data_path = sys.argv[1] #'..\data\region_proposals'
    pass    
    
    
    # inputs, labels = next(iter(train_dataloader))
    # for label in labels:
    #     for ll, label in enumerate(train_dataset.classes):
    #         if ll == train_dataset.class_to_idx: print(label)
    # print(train_dataset.class_to_idx)
    
    
    
    
    
    
    
"""   
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
        DESCRIPTION.bounding_boxs

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





class FramesDataset(Dataset):
    def __init__(self, root_dir, label, transform=None, output = 6, random_seed = 0):
        self.root_dir  = root_dir
        self.transform = transform
        self.label     = label
        self.output    = output
        self.trans_batch = 512
        torch.manual_seed(random_seed)
        self.video_ref   = {}
        self.video_deref = {}
        self.all_items   = []
        self.tensors     = []
        video_hash = 0
        assert label in os.listdir(root_dir), f'folder {label} not found in the root directory'
        
        # creating positive item list
        for i in os.listdir(os.path.join(root_dir, label)):
            img_path = os.path.join(label, i)
            image = plt.imread(os.path.join(self.root_dir,img_path))
            
            self.all_items.append(img_path)
            self.tensors.append(features)
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
            image = plt.imread(os.path.join(self.root_dir,neg_region_name))
            with torch.no_grad():
                features = self.transform(image)
            self.all_items.append(neg_region_name)
            self.tensors.append(features)
            if video_name not in list(self.video_ref):
                self.video_ref[video_name] = video_hash
                self.video_deref[video_hash] = video_name
                video_hash+=1

        self.transform()
        
    def transform(self,):
        tensors = []
        for batch, label in :
            tensors.append(self.transform(batch))
            
    
    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            with torch.no_grad():
                idx = idx.tolist(self.transform(image))
        
        # print(f'getting item {idx} out of {len(self)}')
        item = self.all_items[idx]
        # print(f'item = {item}')
        # img_name = os.path.join(self.root_dir,item)
        # image    = plt.imread(img_name)
        features = self.tensors[idx].squeeze()
        # image    = image.reshape(1,*image.shape)
        label    = torch.tensor(1.) if os.path.split(item)[0]==self.label else torch.tensor(0.)
        video    = torch.tensor(self.video_ref[os.path.split(item)[1].split(';')[1]])
        box      = torch.tensor([int(i) for i in item.split(';')[3:7]])
        frame    = torch.tensor(int(item.split(';')[2]))
        # if self.transform:
        #     with torch.no_grad():
        #         features = self.transform(image)
                # features.requires_grad = False
        if self.output == 6:
            return features, label, idx, box, video, frame
        if self.output == 3:
            return features, label, idx
        return features 

    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    