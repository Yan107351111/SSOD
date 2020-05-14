# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:36:01 2020

@author: yan10
"""
import torch

def Var_multiDim(points):
    '''
    compute the cluster distance variance of each cluster of points.

    Parameters
    ----------
    points : torch.tensor
        tensor of the shape (C, S, D).
        where C is the cluster dimention,
        S is the samples dimention, 
        D is the sample dimention dimention.

    Returns
    -------
    V : torch.tensor
        vector of the variences of the samples in each of the clusters.

    '''
    N   = len(points)
    M1  = torch.mean(points, dim = 0)
    SSE = torch.norm(points-M1, 2, dim = -1)**2
    V   = 1/(N-1)*torch.sum(SSE, dim = -1)
    return V

def Unique(frames):
    '''
    compute the number of uniqe values in each cluster of values.
   
    Parameters
    ----------
    clusters : torch.tensor
        tensor containing the frames of the coresponding samples.
        tensor shape is (C, F),
        where C is as above and F is the frames dimension..

    Returns
    -------
    U : torch.tensor
        vector containing the number of unique frames in each cluster.

    '''
    groups_sorted = torch.sort(frames, -1).values
    sorted_diff   = groups_sorted[:,1:]-groups_sorted[:,:-1]
    U = torch.sum(sorted_diff, -1)+1
    return U

def PotentialScores(features, frames, labels, tau = 50):
    '''
    compute the potential scores of the given clusters.
    
    Parameters
    ----------
    features : torch.tensor
        tensor containing the sample gouped by clusters.
        tensor shape is (C, S, D).
        where C is the cluster dimension,
        S is the samples dimension, 
        D is the sample dimension dimension.
    frames : torch.tensor
        tensor containing the frames of the coresponding samples.
        tensor shape is (C, F),
        where C is as above and F is the frames dimension.
    labels : torch.tensor
        tensor containing the labels of the coresponding samples.
        tensor shape is (C, L),
        where C is as above and L is the labels dimension.
    tau : float, optional
        the softmax temperature. The default is 50.

    Returns
    -------
    S : torch.tensor
        tensor with the shape (C,) containing the Potential Scores.
    '''
    P_k = torch.mean(labels.float(), -1)
    U_k = Unique(frames).float()
    V_k = Var_multiDim(features).float()
    potentials = tau * P_k**2 * torch.log(U_k) / V_k
    S =  torch.softmax(potentials, 0)
    return S



if __name__ == '__main__':
    
    ### Random Demo ###
    torch.manual_seed(0)
    n = 20
    K = 5
    D = 1000
    image_means = torch.rand((K, 1, 1,))
    image_vars  = torch.rand((K, 1, 1,))
    images = torch.rand((K, n, D)) * image_means + image_vars
    
    labels = (torch.rand((K, n,)) < 0.95).int()
    frames = torch.randint(10,(K, n,))
    bbs = torch.cat(
        (
            torch.randint(150,(K, n,2)),
            torch.randint(10,100,(K, n,2))
        ), dim = -1)
    
    
    # tau = 50
    # P_k = torch.mean(labels.float(), -1)
    # U_k = Unique(frames).float()
    # V_k = Var_multiDim(images).float()
    # potentials = tau * P_k**2 * torch.log(U_k) / V_k
    
    print(PotentialScores(images, frames, labels))


