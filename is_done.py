# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 07:41:30 2020

@author: yan10
"""

import os
import sys

if __name__ == '__main__':
    source = sys.argv[1]
    destination = sys.argv[2]
    videos = [i for i in os.listdir(source) if if i[-3:]=='mp4']
    images = [i for i in os.listdir(destination) if i[-3:]=='png']
    combined_destination = ''
    unaccounted = []
    for i in images:
        combined_destination += i+' '
    for video in videos:
        if video in combined_destination:
            continue
        else:
            unaccounted.append(video)
    t = open(sys.argv[3], 'w')
    for video in unaccounted:
        t.write(video)
        t.write('\n')
    t.close()
        
