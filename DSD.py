# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:57:14 2020

@author: yan10
"""

import torch
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : torch.tensor
        dims: 0: points; 1: x, y, w, h
        The x, y position is at the top left corner,
        the w, h dimentions of the box
    bb1 : torch.tensor
        dims: 0: points; 1: x, y, w, h
        The x, y position is at the top left corner,
        the w, h dimentions of the box

    Returns
    -------
    torch.tensor
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    x_left = torch.max(torch.stack((bb1[:,0], bb2[:,0])).T, dim = 1).values
    y_top = torch.max(torch.stack((bb1[:,1], bb2[:,1])).T, dim = 1).values
    x_right = torch.min(torch.stack((bb1[:,0]+bb1[:,2], bb2[:,0]+bb2[:,2])).T, dim = 1).values
    y_bottom = torch.min(torch.stack((bb1[:,1]+bb1[:,3], bb2[:,1]+bb2[:,3])).T, dim = 1).values

    # if x_right < x_left or y_bottom < y_top:
    #     return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    intersection_area = intersection_area*((x_right-x_left)>0)*((y_bottom - y_top)>0)

    # compute the area of both boxes
    bb1_area = bb1[:,2] * bb1[:,3]
    bb2_area = bb2[:,2] * bb2[:,3]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    try:
        iou = intersection_area.float() / (bb1_area + bb2_area - intersection_area)
    except:
        print(f'bb1_area = {bb1_area}')
        print(f'bb2_area = {bb2_area}')
        print(f'intersection_area = {intersection_area}')
        exit(107)
    
    # assert iou >= 0.0
    # assert iou <= 1.0
    return iou

def get_iou_slow_af(bb1, bb2):
    n = len(bb1)
    iou0 = torch.zeros((n,))
    for i in range(n):
        im0 = torch.zeros((260, 260))
        im1 = torch.zeros((260, 260))
        
        im0[bb1[i,0]:bb1[i,0]+bb1[i,2], bb1[i,1]:bb1[i,1]+bb1[i,3]] = 1
        im1[bb2[i,0]:bb2[i,0]+bb2[i,2], bb2[i,1]:bb2[i,1]+bb2[i,3]] = 1
        iou0[i] = torch.sum(im0*im1)/(torch.sum(im0)+torch.sum(im1)-torch.sum(im0*im1))
    return iou0

''' get_iou testing 

n = 20
T = 10000

for i in tqdm(range(T)):
    xy = torch.randint(200,(n,2))
    wh = torch.randint(5,50,(n,2))
    bb1 = torch.cat((xy, wh), dim = 1)
    xy = torch.randint(200,(n,2))
    wh = torch.randint(5,50,(n,2))
    bb2 = torch.cat((xy, wh), dim = 1)
    
    # iou0 = torch.zeros((n,))
    # for i in range(n):
    #     im0 = torch.zeros((260, 260))
    #     im1 = torch.zeros((260, 260))
        
    #     im0[bb1[i,0]:bb1[i,0]+bb1[i,2], bb1[i,1]:bb1[i,1]+bb1[i,3]] = 1
    #     im1[bb2[i,0]:bb2[i,0]+bb2[i,2], bb2[i,1]:bb2[i,1]+bb2[i,3]] = 1
    #     iou0[i] = torch.sum(im0*im1)/(torch.sum(im0)+torch.sum(im1)-torch.sum(im0*im1))


    # x_left = torch.max(torch.stack((bb1[:,0], bb2[:,0])).T, dim = 1).values
    # y_top = torch.max(torch.stack((bb1[:,1], bb2[:,1])).T, dim = 1).values
    # x_right = torch.min(torch.stack((bb1[:,0]+bb1[:,2], bb2[:,0]+bb2[:,2])).T, dim = 1).values
    # y_bottom = torch.min(torch.stack((bb1[:,1]+bb1[:,3], bb2[:,1]+bb2[:,3])).T, dim = 1).values

    # # if x_right < x_left or y_bottom < y_top:
    # #     return 0.0

    # # The intersection of two axis-aligned bounding boxes is always an
    # # axis-aligned bounding box
    # intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # intersection_area = intersection_area*(x_right-x_left>0)*(y_bottom - y_top>0)

    # # compute the area of both boxes
    # bb1_area = bb1[:,2] * bb1[:,3]
    # bb2_area = bb2[:,2] * bb2[:,3]

    # # compute the intersection over union by taking the intersection
    # # area and dividing it by the sum of prediction + ground-truth
    # # areas - the interesection area
    # iou = intersection_area.float() / (bb1_area + bb2_area - intersection_area)
    if (get_iou_slow_af(bb1, bb2)!=0).any():
        print(get_iou_slow_af(bb1, bb2))
        print(get_iou(bb1, bb2))
        break
        
    if (get_iou(bb1, bb2) != get_iou_slow_af(bb1, bb2)).all():
        print()
        print(bb1)
        print(bb2)
        print(get_iou(bb1, bb2))
        print(get_iou_slow_af(bb1, bb2))
        break
'''

def get_graph(bbs, egdes_threshold = 0.5):
    '''
    produce a graph of the provided bounding boxes.
    params:
        bbs:    torch.tensor
                a list of the bounding boxes in a form of [x, y, w, h].
        egdes_threshold:    float if [0,1]
                            The threshold for connecting two bounding boxes
                            with an edge on the graph.
    return:
        graph:  torch.tensor
                a |V|X|V| edge matrix.

    '''
    N = len(bbs)
    # init graph
    # graph = torch.eye(N, dtype = bool)
    graph = torch.zeros((N, N), dtype = bool)
    # compute all graph edges
    bb0s = bbs[[i for j in [[k,]*(N-1-k) for k in range(N)] for i in j]]
    bb1s = bbs[[i for j in range(1,N) for i in range(j,N)]]
    egdes = get_iou(bb0s, bb1s) >= egdes_threshold
    # update graph
    s = 0
    e = N-1
    for i in range(N):
        graph[i, i+1:] = egdes[s:e]
        s = e
        e += N-i-2
    return graph+graph.T

def DSDiscover(graph, V_fraction = 0.1, keys = None):
    '''
    performe DSD on the graph
    params:
        graph:  torch.tensor
                a |V|X|V| edge matrix.
        V_fraction: float if [0,1)
                    the minimal number of vertices to include in the dense
                    subgraph.
        keys:   torch.tensor
                the keys of the provided vertices.
    return:
        V_tag:  torch.tensor
                a vector of the vertices in the dense subgraph.
    '''
    G = graph.clone()+torch.eye(len(graph)).bool()
    DSn = torch.round(torch.tensor([len(G)*0.1])).int().item()
    V_tag = torch.tensor([])
    while len(V_tag) < DSn:
        v_max = torch.sum(G, dim = 1).argmax().reshape(1)
        neighbor = G[v_max].flatten()
        V_neighbor = torch.arange(len(G))[neighbor]
        ### TODO: what is the dense subgraph? V_neighbor or only v_max?
        if keys is None:
            V_tag = torch.cat((V_tag.to(v_max.dtype), v_max)) 
        else:
            V_tag = torch.cat((V_tag.to(keys.dtype), keys[v_max])) 
        G[:,neighbor] = 0
        G[neighbor, :] = 0
    return V_tag

def DSD(bbs, frames):
    '''
    performe DSD on the provided bounding boxes according to the frames they
    were taken from.
    params:
        bbs:    torch.tensor
                a Nx4 list of bounding boxes in the format of [x, y, w, h].
        frames: torch.tensor
                a N long vector of indicators to the frames the coresponding
                bounding boxes were taken from.
    return:
        DS:     torch.tensor
                a vector of the indices of the bounding boxed chosen as
                vertices in the dense subgraphs.
    '''
    frame_sets = torch.tensor(list(set(frames.tolist())))
    indices    = torch.arange(len(frames))
    DSs        = torch.tensor([])
    for frame in frame_sets:
        # fetch the bounding boxed in the frame
        bbs_frm = bbs[frames==frame]
        graph = get_graph(bbs_frm)
        DS  = DSDiscover(graph, keys = indices[frames==frame])
        DSs = torch.cat((DSs.to(DS.dtype), DS)) 
    return DSs
    

if __name__ == '__main__':
    torch.manual_seed(0)
    n = 300
    images = torch.randint(2,(n,1))
    labels = torch.randint(2,(n,))
    frames = torch.randint(1,(n,))
    bbs = torch.cat(
        (
            torch.randint(150,(n,2)),
            torch.randint(10,100,(n,2))
        ), dim = 1)
    
    samples = (images, labels, frames, bbs) 
        
    frame_sets = torch.tensor(list(set(frames.tolist())))
    for frame in frame_sets:
        # fetch the bounding boxed in the frame
        bbs_frm = bbs[frames==frame]
        graph = get_graph(bbs_frm)
        # plt.figure()
        # plt.imshow(graph)
        dsd = DSDiscover(graph)
        
        
    
    
    




















