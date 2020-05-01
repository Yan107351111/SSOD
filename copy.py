# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:15:54 2020

@author: yan10
"""
import os
import shutil
os.mkdir('example')
l = [i for i in os.listdir('frames') if i[:3]!='neg']
j = 0
while j<100:
    if l[-j][:3]!='neg':
        j+=1
        print(j)
        shutil.copyfile(os.path.join('frames',l[-j]), os.path.join('example',l[-j]))
