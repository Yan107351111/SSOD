# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:37:11 2020

@author: yan10
"""

import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from tqdm import tqdm
import time

if __name__ == '__main__':
    
    ### ARGS ###
    DATA_PATH = '..\\..\\data\\example'
    OUT_PATH  = '..\\..\\data\\region_proposals'
    ############ defults
    N      = 300 # number of reg props
    SKIP   = 5   # take only SKIP'th proposal
    LABELS = ['bike', 'cup', 'dog', 'drum', 'guitar',
               'gun', 'horse', 'pan', 'plate',
               'scissors', 'tire']
    IMSIZE = 299 # 224 fml
    ############
    
    image_names = [image_name
                   for image_name in os.listdir(DATA_PATH)
                   if not image_name.startswith('neg')]
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    try:
        os.mkdir(OUT_PATH)
    except:pass
    for label in LABELS:
        try:
            os.mkdir(os.path.join(OUT_PATH, label))
        except:pass 
    count_total = len(image_names)
    count       = 0
    start_time  = time.time()
    for image_name in tqdm(image_names):
        count+=1
        if count%500==0:
            print(f'runing time:{time.time()-start_time}')
            print(f'processed: {count}/{count_total}')
            print(f'remaining time approximately: {(time.time()-start_time)/count*(count_total-count)}')
        for label in LABELS:
            if image_name.startswith(label):
                break
        # process image for reg props
        image = cv2.imread(os.path.join(DATA_PATH, image_name))
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        
        ims    = 0
        for e,result in enumerate(ssresults):
            
            if e%SKIP == 0 and ims<N:
                x,y,w,h = result
                crop = image[y:y+h, x:x+w]
                
                cv2.imwrite(
                    os.path.join(
                        OUT_PATH,
                        label,
                        f'{image_name[:-4]};{x};{y};{w};{h};{ims:04}.png',
                        ),
                    cv2.resize(crop, (IMSIZE, IMSIZE))
                    )
                
                ims+=1            
        
    
    
    
    
    




