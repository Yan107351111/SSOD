# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:37:11 2020

@author: yan10
"""
import cv2
# from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
import time
import torch
from tqdm import tqdm
from typing import Union, Tuple


def selective_search(
        data_path: str, out_path: str, label: str = None,
        region_num: int = 2000, region_skip: int = 2,
        imsize: Union[int, Tuple] = 299,
        min_width: int = 80, min_hight: int = 80,
        min_size: int = 200, max_dim_ratio: float = 4,
        to_file = True, silent = False):
    '''
    perform selective search on images in folder "data_path".
    output rescaled to "imsize" images will be saved at "out_path".

    Parameters
    ----------
    data_path : str
        path leading to data images.
    out_path : str
        path the region proposal will be saved to.
    label : str
        the label of positive image set.
    region_num : int, optional
        Number of total region proposals to take from each image.
        The default is 2000.
    region_skip : int, optional
        The number of region proposals to skip for ever saved one.
        The default is 2.
    imsize : Union[int, Tuple], optional
        the size of the resized region proposals.
        The default is 299.
    min_width : int, optional
        the minimal output image width.
        The default is 80.
    min_hight : int, optional
        the minimal output image hight.
        The default is 80.
    min_size : int, optional
        the minimal output image size in pixels.
        The default is 200.
    max_dim_ratio : float, optional
        the maximal allowed ratio between the output image
        hight and width.
        The default is 4.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.
    
    "Given an image, about 2, 000 object proposals from Selective
    Search [20] or EdgeBox [47] are generated" 
    - https://arxiv.org/pdf/1807.03342.pdf @ 3. Method
    '''
    if  isinstance(imsize, int):
        imsize = (299, 299)
    elif isinstance(imsize, tuple):
        pass
    else:
        raise ValueError(f'imsize not recognised\n'
                         'Supposed to be one of: int, tuple\n'
                         'But got {type(imsize)}')
    assert region_skip>=0
    
    if os.path.isfile(data_path):
        image_names = [data_path]
        single = True
    else:
        image_names = [image_name
                       for image_name in os.listdir(data_path)]
        single = False
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    try:
        os.mkdir(out_path)
    except:pass
    
    # count_total = len(image_names)
    # count       = 0
    # start_time  = time.time()
    names = []
    regions = []
    bounding_boxs = []
    for image_name in tqdm(image_names, disable = silent):
        # count+=1
        # if count%500==0:
        #     print(f'runing time:{time.time()-start_time}')
        #     print(f'processed: {count}/{count_total}')
        #     print(f'remaining time approximately: {(time.time()-start_time)/count*(count_total-count)}')
        # process image for reg props
        if single: image = cv2.imread(image_name)
        else: image = cv2.imread(os.path.join(data_path, image_name))
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        
        ims  = 0
        for init_skip in range(region_skip+1):
            if ims>=region_num:
                    break
            skip = init_skip
            for e,result in enumerate(ssresults):
                if skip==init_skip+region_skip+1:
                    skip = init_skip
                if skip!=init_skip:
                    skip+=1
                    continue
                x,y,w,h = result
                # if the region is not up to standards, skip it
                if w<min_width or h<min_hight or w*h<min_size \
                   or float(w)/float(h) > max_dim_ratio       \
                   or float(w)/float(h) < 1/max_dim_ratio:
                    skip+=1
                    continue
                # crop, resize and save
                crop = image[y:y+h, x:x+w]
                region = cv2.resize(crop, imsize, interpolation = cv2.INTER_AREA)
                file_name = f'{image_name[:-4]};{x};{y};{w};{h};{ims:04}.png'
                if to_file:
                    cv2.imwrite(
                        os.path.join(
                            out_path,
                            file_name,
                            ),
                        region
                        )
                else:
                    names.append(file_name)
                    regions.append(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                    bounding_boxs.append(torch.tensor([x,y,w,h]))
                ims+=1            
                skip+=1
                if ims>=region_num:
                    break
        if not to_file:
            return names, regions, torch.stack(bounding_boxs)

        


if __name__ == '__main__':
    
    ### ARGS ###
    DATA_PATH = sys.argv[1]#'..\\..\\data\\example'
    OUT_PATH  = sys.argv[2]#'..\\..\\data\\region_proposals'
    ############ defults
    N      = 2000 # number of reg props
    SKIP   = 1   # take only SKIP'th proposal
    LABELS = ['bike', 'cup', 'dog', 'drum', 'guitar',
               'gun', 'horse', 'pan', 'plate',
               'scissors', 'tire']
    IMSIZE = 299
    MIN_W = 30
    MIN_H = 30
    MIN_P = 200
    ############
    
    selective_search(
        DATA_PATH, OUT_PATH, LABELS,
        N, IMSIZE, MIN_W, MIN_H, MIN_P,
    )
    
            
                    
                    
        
    
    
    
    
    




