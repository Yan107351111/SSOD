#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:38:09 2020

@author: yanivzis@bm.technion.ac.il
"""


import sys
import datetime
import os
from SelectiveSearch import selective_search
import shutil
import torch





if __name__ == '__main__':
    
    data_path = sys.argv[1]
    label = sys.argv[2]
    out_dir = sys.argv[3]
    if len(sys.argv)>5:
        region_num  = sys.argv[4]
        region_skip = sys.argv[5]
    else:
        region_num  = 2000
        region_skip = 2
    torch.manual_seed(42)
    
    dfname = f'dataset_{label}_'+str(datetime.date.today()).replace('-', '_')
    try: os.mkdir(os.path.join(out_dir, dfname))
    except: pass
    try: os.mkdir(os.path.join(out_dir, dfname, 'positive'))
    except: pass
    try: os.mkdir(os.path.join(out_dir, dfname, 'negative'))
    except: pass

    
    positive_images = [i for i in os.listdir(os.path.join(data_path, label))]
    other_labels = [i for i in os.listdir(data_path) if i!=label]
    nagetive_images = []
    i = 0
    while i < len(positive_images):
        l = torch.randint(len(other_labels), (1,)).item()
        negatives = [n for n in os.listdir(os.path.join(data_path, other_labels[l]))]
        n = torch.randint(len(negatives), (1,)).item()
        negative = os.path.join(data_path, other_labels[l], negatives[n])
        if negative not in nagetive_images:
            nagetive_images.append(negative)
            i+=1
    
    
    for img in positive_images:
        shutil.copyfile(
            os.path.join(data_path, label, img),
            os.path.join(out_dir, dfname, 'positive', img)
        )
        
    for img in nagetive_images:
        shutil.copyfile(
            os.path.join(img),
            os.path.join(out_dir, dfname, 'negative', img.split('/')[-1])
        )

    region_floder = os.path.join(out_dir, dfname, 'regions')
    
    selective_search(os.path.join(out_dir, dfname, 'positive'), region_floder, region_num = 50, region_skip = 10)
    selective_search(os.path.join(out_dir, dfname, 'negative'), region_floder, region_num = 50, region_skip = 10)



