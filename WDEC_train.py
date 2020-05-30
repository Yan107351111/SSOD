# -*- coding: utf-8 -*- 
"""
Created on Thu May 21 17:23:14 2020

@author: yan10
"""

# from FeatureExtraction import get_dataset
from DSD import DSD
import numpy as np

from PotentialScoring import PotentialScores

# from ptsdae.sdae import StackedDenoisingAutoEncoder
# import ptsdae.model as ae
# from ptdec.dec import WDEC
from ptdec.model import predict
from ptdec.utils import target_distribution, cluster_accuracy

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
# from torch.optim import SGD
# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader, default_collate


from tqdm import tqdm
from typing import Tuple, Callable, Optional, Union

def SSKMeans(
        wdec: torch.nn.Module,
        features: torch.tensor,
        actual: torch.tensor,
        idxs: torch.tensor,
        boxs: torch.tensor,
        videos: torch.tensor,
        frames: torch.tensor,
    ):
    '''
    

    Parameters
    ----------
    wdec : torch.nn.Module
        DESCRIPTION.
    features : torch.tensor
        DESCRIPTION.
    actual : torch.tensor
        DESCRIPTION.
    idxs : torch.tensor
        DESCRIPTION.
    boxs : torch.tensor
        DESCRIPTION.
    videos : torch.tensor
        DESCRIPTION.
    frames : torch.tensor
        DESCRIPTION.

    Returns
    -------
    predicted : np.ndarray
        DESCRIPTION.
    kmeans : KMeans
        DESCRIPTION.
    '''
    # print('\n\n\n')
    # print('performing KMeans')
    # print('\n\n\n')
    
    kmeans = KMeans(n_clusters=wdec.cluster_number, n_init=20)
    # compute the weighted K-Means sample weights using potential scores and
    # DSD
    if wdec.assignment.cluster_predicted is not None:
        feature_list  = []
        video_list    = []
        label_list    = []
        K             = wdec.assignment.cluster_number
        idxs, indices = torch.sort(idxs)
        features      = features[indices]
        actual        = actual[indices]
        boxs          = boxs[indices]
        videos        = videos[indices]
        frames        = frames[indices]
        DCD_count     = torch.zeros((K,))
        for C in range(K):
            #print('\n\n\n')
            #print(f'wdec.assignment.cluster_predicted.shape = {wdec.assignment.cluster_predicted}')
            #print('\n\n\n')
            C_bool = wdec.assignment.cluster_predicted[:,1]==C
            C_inds = wdec.assignment.cluster_predicted[:,0][C_bool].long()
            feature_list.append(features[C_inds])
            video_list.append(videos[C_inds])
            label_list.append(actual[C_inds])
            video_frames = videos[C_inds]*10000 + frames[C_inds]
            ## Run DSD
            DCD_count[C] = len(DSD(boxs[C_inds], video_frames))
            
        ## Compute the potential score Sk in (1) for each cluster
        ## set τ = 50
        sample_weights = PotentialScores(
            feature_list, video_list, label_list,
        ) 
        if (sample_weights!=sample_weights).any():
            raise ValueError(f'Self similarity test failure 0\nsample_weights = {sample_weights}')
        sample_weights[DCD_count>0] /= DCD_count[DCD_count>0] 
        if (sample_weights!=sample_weights).any():
            raise ValueError(f'Self similarity test failure 1\n'
                             f'sample_weights = {sample_weights}'
                             f'DCD_count = {DCD_count}')
        _, pred_idx    = wdec.assignment.cluster_predicted[:,0].sort()
        custer_idx     = wdec.assignment.cluster_predicted[pred_idx,1].long()
        sample_weights = sample_weights[custer_idx]
        # TODO: find out what "normalized by the number of positive samples
        # in the cluster defined by DSD" means (question no.10 in notebook)
    else: sample_weights = None
    #print(features.shape)
    ## Re-initialize cluster centers using Weighted K-Means
    # print('\n\n\n')
    # print(f'sample_weights = {sample_weights}')
    # print('\n\n\n')
    predicted = kmeans.fit_predict(
        features.numpy(),
        sample_weight = sample_weights, # model.assignment.weights(actual, idxs),
    )
    return predicted, kmeans
    

def PositiveRatioClusters(
        predicted: np.ndarray,
        actual: torch.tensor,
        K: int,
    ) -> torch.tensor:
    '''
    TODO:

    Parameters
    ----------
    predicted : np.ndarray
        DESCRIPTION.
    actual : torch.tensor
        DESCRIPTION.
    K : int
        DESCRIPTION.

    Returns
    -------
    cp_freq : torch.tensor
        DESCRIPTION.

    '''
    pred_rep  = torch.tensor(predicted).repeat(1,K).reshape(K,-1)
    c_rep     = (pred_rep == torch.arange(K).reshape(-1,1)).int()
    c_sizes   = c_rep.sum(-1)
    cp_rep    = c_rep*actual.repeat((K,1))
    cp_sizes  = cp_rep.sum(-1)
    cp_freq   = cp_sizes/c_sizes
    # tensor [Clusters]. 1 if the cluster is a positive ratio cluster
    # and 0 otherwise.
    return cp_freq

def ReInitKMeans(wdec, data_iterator): # TODO: finish function and put in train function
    raise NotImplementedError()


def DataSetExtract(
        dataset: torch.utils.data.Dataset,
        wdec: torch.nn.Module = None,
        silent: bool = False,
        batch_size: int = 512,
        collate_fn = default_collate,
        sampler: Optional[torch.utils.data.sampler.Sampler] = None,
        cuda: bool = True,
    ):
    '''
    TODO:

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        DESCRIPTION.
    wdec : torch.nn.Module
        DESCRIPTION.
    silent : bool, optional
        DESCRIPTION. The default is False.
    batch_size : int, optional
        DESCRIPTION. The default is 512.
    collate_fn : TYPE, optional
        DESCRIPTION. The default is default_collate.
    sampler : Optional[torch.utils.data.sampler.Sampler], optional
        DESCRIPTION. The default is None.
    cuda : bool, optional
        DESCRIPTION. The default is True.
     : TYPE
        DESCRIPTION.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    features : TYPE
        DESCRIPTION.
    actual : TYPE
        DESCRIPTION.
    idxs : TYPE
        DESCRIPTION.
    boxs : TYPE
        DESCRIPTION.
    videos : TYPE
        DESCRIPTION.

    '''
    static_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        sampler=sampler,
        shuffle=False
    )
    data_iterator = tqdm(
        static_dataloader,
        leave=True,
        unit='batch',
        postfix={
            'epo': -1,
            'acc': '%.4f' % 0.0,
            'lss': '%.8f' % 0.0,
            'dlb': '%.4f' % -1,
        },
        disable=silent
    )
    features = []
    actual   = []
    idxs     = []
    videos   = []
    boxs     = []
    frames   = []
    # form initial cluster centres
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) > 3:
            batch, label, idx, box, video, frame = batch  # if we have a prediction label, separate it to actual
            actual.append(label)
            idxs.append(idx)
            boxs.append(box)
            videos.append(video)
            frames.append(frame)
        else: raise RuntimeError('Dataset is\'nt providing all necessary information: batch, label, idx, box, video')
        if cuda:
            batch = batch.cuda(non_blocking = True)
        if wdec is not None:
            features.append(wdec.encoder(batch).detach().cpu())
        else:
            features.append(batch)
    features  = torch.cat(features)
    actual    = torch.cat(actual).long()
    idxs      = torch.cat(idxs).long()
    boxs      = torch.cat(boxs).long()
    videos    = torch.cat(videos).long()
    frames    = torch.cat(frames).long()
    return features, actual, idxs, boxs, videos, frames


def train(dataset: torch.utils.data.Dataset,
          wdec: torch.nn.Module,
          epochs: int,
          batch_size: int,
          optimizer: torch.optim.Optimizer,
          reinitKMeans: bool = True,
          scheduler = None, ###
          positive_ratio: float = 0.6, ###
          stopping_delta: Optional[float] = None,
          collate_fn = default_collate,
          cuda: bool = True,
          sampler: Optional[torch.utils.data.sampler.Sampler] = None,
          silent: bool = False,
          update_freq: int = 10,
          evaluate_batch_size: int = 1024,
          update_callback: Optional[Callable[[float, float], None]] = None,
          epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None) -> None:
    """
    Train the DEC model given a dataset, a model instance and various configuration parameters.

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to train
    :param epochs: number of training epochs
    :param batch_size: size of the batch to train with
    :param reinitKMeans: if true, the clusters will be initialized.
    :param optimizer: instance of optimizer to use
    :param scheduler: instance of lr_scheduler to use
    :param stopping_delta: label delta as a proportion to use for stopping, None to disable, default None
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether to use CUDA, defaults to True
    :param sampler: optional sampler to use in the DataLoader, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, None disables, default 10
    :param evaluate_batch_size: batch size for evaluation stage, default 1024
    :param update_callback:sample_weight optional function of accuracy and loss to update, default None
    :param epoch_callback: optional function of epoch and model, default None
    :return: None
    """
    static_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        sampler=sampler,
        shuffle=False
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=True
    )
    data_iterator = tqdm(
        static_dataloader,
        leave=True,
        unit='batch',
        postfix={
            'epo': -1,
            'acc': '%.4f' % 0.0,
            'lss': '%.8f' % 0.0,
            'dlb': '%.4f' % -1,
        },
        disable=silent
    )
    wdec.train()
    
    if reinitKMeans:
        # get all data needed for KMeans.
        features, actual, idxs, boxs, videos, frames = DataSetExtract(dataset, wdec)
        # KMeans.
        predicted, kmeans = SSKMeans(
            wdec, features, actual, idxs, boxs, videos, frames
        )
        # Computing the positive ration scores and the positive ratio clusters
        cpr = PositiveRatioClusters(
            predicted, actual, wdec.assignment.cluster_number,
        )
        predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
        _, accuracy        = cluster_accuracy(predicted, actual.cpu().numpy())
        cluster_centers    = torch.tensor(
            kmeans.cluster_centers_,
            dtype=torch.float, requires_grad=True
        )
        predicted_idxed    = torch.cat(
            [idxs.reshape(-1,1), torch.tensor(predicted).reshape(-1,1).long()],
            dim = -1
        )
        if cuda:
            cluster_centers = cluster_centers.cuda(non_blocking=True)
        with torch.no_grad():
            # initialise the cluster centers
            wdec.state_dict()['assignment.cluster_centers'].copy_(cluster_centers)
            # wdec.state_dict()['assignment.cluster_predicted'].copy_(predicted_idxed)
            # wdec.state_dict()['assignment.cluster_positive_ratio'].copy_(cpr)
            wdec.assignment.cluster_predicted = predicted_idxed.clone()
            wdec.assignment.cluster_positive_ratio = cpr.clone()
    else:
      predicted, actual = predict(
            dataset,
            wdec,
            batch_size=evaluate_batch_size,
            collate_fn=collate_fn,
            silent=True,
            return_actual=True,
            cuda=cuda
        )
      predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
      _, accuracy = cluster_accuracy(predicted.cpu().numpy(), actual.cpu().numpy())
        
    loss_function = nn.KLDivLoss(size_average=False)
    delta_label = None
    for epoch in range(epochs):
        # features = [] ### I see no use for this
        data_iterator = tqdm(
            train_dataloader,
            leave=True,
            unit='batch',
            postfix={
                'epo': epoch,
                'acc': '%.4f' % (accuracy or 0.0),
                'lss': '%.8f' % 0.0,
                'dlb': '%.4f' % (delta_label or 0.0),
            },
            disable=silent,
        )
        wdec.train()
        for index, batch in enumerate(data_iterator):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 6:
                batch, actual, idxs, _, _, _ = batch  # if we have a prediction label, strip it away
            if cuda:
                batch  = batch.cuda(non_blocking=True)
                actual = actual.cuda()
                idxs   = idxs.cuda()
            output = wdec(batch, actual, idxs,)
            target = target_distribution(output).detach()
            loss   = loss_function(output.log(), target) / output.shape[0]
            data_iterator.set_postfix(
                epo = epoch,
                acc = '%.4f' % (accuracy or 0.0),
                lss = '%.8f' % float(loss.item()),
                dlb = '%.4f' % (delta_label or 0.0),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            if scheduler is not None: scheduler.step()
            # features.append(model.encoder(batch).detach().cpu()) ### I see no use for this
            if update_freq is not None and index % update_freq == 0:
                loss_value = float(loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    acc='%.4f' % (accuracy or 0.0),
                    lss='%.8f' % loss_value,
                    dlb='%.4f' % (delta_label or 0.0),
                )
                if update_callback is not None:
                    update_callback(accuracy, loss_value, delta_label)
        predicted, actual = predict(
            dataset,
            wdec,
            batch_size=evaluate_batch_size,
            collate_fn=collate_fn,
            silent=True,
            return_actual=True,
            cuda=cuda
        )
        delta_label = float((predicted != predicted_previous).float().sum().item()) / predicted_previous.shape[0]
        if stopping_delta is not None and delta_label < stopping_delta:
            print('Early stopping as label delta "%1.5f" less than "%1.5f".' % (delta_label, stopping_delta))
            break
        predicted_previous = predicted
        _, accuracy = cluster_accuracy(predicted.cpu().numpy(), actual.cpu().numpy())
        data_iterator.set_postfix(
            epo=epoch,
            acc='%.4f' % (accuracy or 0.0),
            lss='%.8f' % 0.0,
            dlb='%.4f' % (delta_label or 0.0),
        )
        if epoch_callback is not None:
            epoch_callback(epoch, wdec)


def predict(dataset: torch.utils.data.Dataset,
            model: torch.nn.Module,
            batch_size: int = 1024,
            collate_fn = default_collate,
            cuda: bool = True,
            silent: bool = False,
            return_actual: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Predict clusters for a dataset given a DEC model instance and various configuration parameters.

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to predict
    :param batch_size: size of the batch to predict with, default 1024
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param return_actual: return actual values, if present in the Dataset
    :return: tuple of prediction and actual if return_actual is True otherwise prediction
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    data_iterator = tqdm(
        dataloader,
        leave=True,
        unit='batch',
        disable=silent,
    )
    features = []
    actual = []
    model.eval()
    for batch in data_iterator:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 6:
            batch, value, idxs, _, _, _ = batch  # unpack if we have a prediction label
            if return_actual:
                actual.append(value)
        elif return_actual:
            raise ValueError('Dataset has no actual value to unpack, but return_actual is set.')
        if cuda:
            batch = batch.cuda(non_blocking=True)
            value = value.cuda()
            idxs   = idxs.cuda()            
        features.append(model(batch, value, idxs,).detach().cpu())  # move to the CPU to prevent out of memory on the GPU
    if return_actual:
        return torch.cat(features).max(1)[1], torch.cat(actual).long()
    else:
        return torch.cat(features).max(1)[1]



'''
class PotentialSampler(Sampler):
    def __init__(
            self,
            data_source: torch.utils.data.Dataset,
            positive_idxs: torch.Tensor,
            semple_distribution: torch.Tensor,
            num_samples: int,
        ):
        
        self.data_source = data_source
        self.num_samples = num_samples
        self.positive_idxs = positive_idxs
        negative_idxs = []
        for i in range(num_samples):
            if (positive_idxs!=i).any():
                negative_idxs.append(i)
        self.negative_idxs = torch.tensor(negative_idxs)
        self.semple_distribution = semple_distribution

    def __iter__(self):
        
        # sample from uniform distribution.
        rnd = torch.rand(self.num_samples)
        inds = torch.zeros_like(rnd)
        # transform half of the pdf to uniformly sampled negative samples.
        inds[rnd<0.5] = (rnd[rnd<0.5]*2*len(self.negative_idxs)+1).int()
        # compute the positive sample cdf.
        thresholds = self.semple_distribution.cumsum(0).reshape(-1,1)
        # transform half of the initial pdf to samples from the positive
        # sample distribution.
        positives = ((rnd[rnd>=0.5]-0.5)*2>thresholds).sum(0)
        inds[rnd>=0.5] = self.positive_idxs[positives]
        return iter(range(len(self.data_source)))
    
    def __len__(self):
        return len(self.data_source)
'''












