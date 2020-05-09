# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:35:33 2020

@author: yan10
"""


from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from FeatureExtraction import get_dataloader, get_dataset




pretrain_epochs   = 300
finetune_epochs   = 500
training_callback = None
cuda         = torch.cuda.is_available()
batch_size   = 256
data_path    = ''
ds_train     = get_dataset(data_path, batch_size)
ds_val       = 0
embedded_dim = 1000
autoencoder  = StackedDenoisingAutoEncoder(
        [embedded_dim, 500, 500, 2000, 10],
        final_activation=None
    )
if cuda:
    autoencoder.cuda()

print('Pretraining stage.')
ae.pretrain(
    ds_train,
    autoencoder,
    cuda       = cuda,
    validation = ds_val,
    epochs     = pretrain_epochs,
    batch_size = batch_size,
    optimizer  = lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
    scheduler  = lambda x: StepLR(x, 100, gamma=0.1),
    corruption = 0.2
)

print('Training stage.')
ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
ae.train(
    ds_train,
    autoencoder,
    cuda=cuda,
    validation=ds_val,
    epochs=finetune_epochs,
    batch_size=batch_size,
    optimizer=ae_optimizer,
    scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
    corruption=0.2,
    update_callback=training_callback
)























