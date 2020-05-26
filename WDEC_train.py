# -*- coding: utf-8 -*- 
"""
Created on Thu May 21 17:23:14 2020

@author: yan10
"""

from FeatureExtraction import get_dataset
import numpy as np

from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.dec import DEC
from ptdec.model import train
from ptdec.utils import target_distribution, cluster_accuracy

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader, default_collate

from tqdm import tqdm
from typing import Tuple, Callable, Optional, Union


def train(dataset: torch.utils.data.Dataset,
          model: torch.nn.Module,
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
    :param update_callback: optional function of accuracy and loss to update, default None
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
    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
    model.train()
    features = []
    actual = []
    # form initial cluster centres
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, label, _, _ = batch  # if we have a prediction label, separate it to actual
            actual.append(label)
        if cuda:
            batch = batch.cuda(non_blocking = True)
        features.append(model.encoder(batch).detach().cpu())
    actual    = torch.cat(actual).long()
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    pred_rep  = torch.tensor(predicted).repeat(1,50).reshape(50,-1)
    c_rep     = (pred_rep == torch.arange(50).reshape(-1,1)).int()
    c_sizes   = c_rep.sum(-1)
    cp_rep    = c_rep*actual.repeat((50,1))
    cp_sizes  = cp_rep.sum(-1)
    cp_freq   = cp_sizes/c_sizes
    cpr       = cp_freq > positive_ratio
    # predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    # _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)
    if cuda:
        cluster_centers = cluster_centers.cuda(non_blocking=True)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()['assignment.cluster_centers'].copy_(cluster_centers)
        model.state_dict()['assignment.cluster_predicted'].copy_(cluster_centers)
        model.state_dict()['assignment.cluster_positive_ratio'].copy_(cpr)
    loss_function = nn.KLDivLoss(size_average=False)
    delta_label = None
    for epoch in range(epochs):
        features = []
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
        model.train()
        for index, batch in enumerate(data_iterator):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, _ = batch  # if we have a prediction label, strip it away
            if cuda:
                batch = batch.cuda(non_blocking=True)
            output = model(batch)
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
            features.append(model.encoder(batch).detach().cpu())
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
            model,
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
            epoch_callback(epoch, model)















