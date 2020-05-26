# -*- coding: utf-8 -*- 
"""
Created on Thu May 21 17:23:14 2020

@author: yan10
"""

# from FeatureExtraction import get_dataset
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
from torch.utils.data.dataloader import DataLoader, default_collate

from tqdm import tqdm
from typing import Tuple, Callable, Optional, Union


def train(dataset: torch.utils.data.Dataset,
          wdec: torch.nn.Module,
          detector: torch.nn.Module,
          epochs: int,
          batch_size: int,
          optimizer: torch.optim.Optimizer,
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
    :param optimizer: instance of optimizer to use
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
    kmeans = KMeans(n_clusters=wdec.cluster_number, n_init=20)
    wdec.train()
    features = []
    actual   = []
    idxs     = []
    videos   = []
    boxs     = []
    # form initial cluster centres
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) > 3:
            batch, label, idx, box, video = batch  # if we have a prediction label, separate it to actual
            actual.append(label)
            idxs.append(idx)
            boxs.append(box)
            videos.append(video)
        else: raise RuntimeError('Dataset is\'nt providing all necessary information: batch, label, idx, box, video')
        # elif (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) > 1:
        #     batch, label, idx = batch  # if we have a prediction label, separate it to actual
        #     actual.append(label)
        #     idxs.append(idx)
        if cuda:
            batch = batch.cuda(non_blocking = True)
        features.append(wdec.encoder(batch).detach().cpu())
    features  = torch.cat(features)
    actual    = torch.cat(actual).long()
    idxs      = torch.cat(idxs).long()
    boxs      = torch.cat(boxs).long()
    videos    = torch.cat(videos).long()
    
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
        for C in range(K):
            C_bool = wdec.assignment.cluster_predicted[:,1]==C
            C_inds = wdec.assignment.cluster_predicted[:,0][C_bool]
            feature_list.append(features[C_inds])
            video_list.append(videos[C_inds])
            label_list.append(actual[C_inds])
        sample_weights = PotentialScores(
            feature_list, video_list, label_list,
        ) 
    else: sample_weights = None
    
    predicted = kmeans.fit_predict(
        features.numpy(),
        sample_weight = sample_weights, # model.assignment.weights(actual, idxs),
    )
    
    # Computing the positive ration scores and the positive ratio clusters
    pred_rep  = torch.tensor(predicted).repeat(1,50).reshape(50,-1)
    c_rep     = (pred_rep == torch.arange(50).reshape(-1,1)).int()
    c_sizes   = c_rep.sum(-1)
    cp_rep    = c_rep*actual.repeat((50,1))
    cp_sizes  = cp_rep.sum(-1)
    cp_freq   = cp_sizes/c_sizes
    # tensor [Clusters]. 1 if the cluster is a positive ratio cluster
    # and 0 otherwise.
    cpr       = cp_freq > positive_ratio
    
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    _, accuracy        = cluster_accuracy(predicted, actual.cpu().numpy())
    cluster_centers    = torch.tensor(
        kmeans.cluster_centers_,
        dtype=torch.float, requires_grad=True
    )
    predicted_idxed    = torch.cat(
        [idxs.reshape(-1,1), predicted.reshape(-1,1)],
        dim = -1
    )
    if cuda:
        cluster_centers = cluster_centers.cuda(non_blocking=True)
    with torch.no_grad():
        # initialise the cluster centers
        wdec.state_dict()['assignment.cluster_centers'].copy_(cluster_centers)
        wdec.state_dict()['assignment.cluster_predicted'].copy_(predicted_idxed)
        wdec.state_dict()['assignment.cluster_positive_ratio'].copy_(cpr)
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
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, _ = batch  # if we have a prediction label, strip it away
            if cuda:
                batch = batch.cuda(non_blocking=True)
            output = wdec(batch)
            target = target_distribution(output).detach()
            loss = loss_function(output.log(), target) / output.shape[0]
            data_iterator.set_postfix(
                epo=epoch,
                acc='%.4f' % (accuracy or 0.0),
                lss='%.8f' % float(loss.item()),
                dlb='%.4f' % (delta_label or 0.0),
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















