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

### ARGS ###
DATA_PATH = sys.argv[1]#'..\\..\\data\\example'
OUT_PATH  = sys.argv[2]#'..\\..\\data\\region_proposals'
############ defults
N      = 300 # number of reg props
SKIP   = 1   # take only SKIP'th proposal
LABELS = ['bike', 'cup', 'dog', 'drum', 'guitar',
           'gun', 'horse', 'pan', 'plate',
           'scissors', 'tire']
IMSIZE = 224
############


image_names = os.listdir(DATA_PATH)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
try:
    os.mkdir(OUT_PATH)
except:pass
for label in LABELS+['neg',]:
    try:
        os.mkdir(os.path.join(OUT_PATH, label))
    except:pass 
for image_name in image_names:
    for label in LABELS+['neg',]:
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
                    f'{image_name[:-4]}_{ims:04}.png',
                    ),
                cv2.resize(crop, (IMSIZE, IMSIZE))
                )
            
            ims+=1            
    
    # fig = plt.figure(figsize=(4., 4.))
    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                  nrows_ncols=(10, 10),  # creates 2x2 grid of axes
    #                  axes_pad=0.1,  # pad between axes in inch.
    #                  )
    # for ax, im in zip(grid, imlist):
    #     # Iterating over the grid returns the Axes.
    #     ax.imshow(im)
    # plt.show()
    
     












