import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Tuple, Callable, Optional, Union
from DSD import DSD

class ClusterAssignment(nn.Module):
    def __init__(self,
                 cluster_number: int,
                 embedding_dimension: int,
                 alpha: float = 1.0,
                 cluster_centers: Optional[torch.Tensor] = None) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number,
                self.embedding_dimension,
                dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class WeightedClusterAssignment(nn.Module):
    def __init__(self,
                 cluster_number: int,
                 embedding_dimension: int,
                 positive_ratio_threshold: float = 0.6,
                 alpha: float = 1.0,
                 cluster_centers: Optional[torch.Tensor] = None) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param positive_ratio_threshold: positive ration above which sample weighting will be performed.
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(WeightedClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.positive_ratio_threshold = Parameter(torch.tensor(positive_ratio_threshold))
        self.alpha = Parameter(torch.tensor(alpha))
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number,
                self.embedding_dimension,
                dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)
        self.cluster_predicted = None
        self.cluster_positive_ratio = None

    def forward(self, batch: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :param labels: FloatTensor of size [batch size]
        :param idx: FloatTensor of size [batch size]
        :return: FloatTensor [batch size, number of clusters]
        """
        # print('\n\n\n')
        # print(f'batch.unsqueeze(1).shape = {batch.unsqueeze(1).shape}')
        # print(f'self.cluster_centers.shape = {self.cluster_centers.shape}')
        # print(f'self.weights(labels, idx).shape = {self.weights(labels, idx).shape}')
        # print('\n\n\n')
        norm_squared = torch.sum(
            (batch.unsqueeze(1) - self.cluster_centers) ** 2
            * self.weights(labels, idx).reshape(-1,1,1),
            2
        )
        if (norm_squared!=norm_squared).any(): raise ValueError('nan found at norm_squared')
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        if (norm_squared!=norm_squared).any(): raise ValueError('nan found at numerator0')
        power = float(self.alpha + 1) / 2
        numerator = numerator**power 
        if (norm_squared!=norm_squared).any(): raise ValueError('nan found at numerator1')
        denumerator = torch.sum(numerator, dim=1, keepdim=True)
        if (denumerator!=denumerator).any(): raise ValueError(f'nan found at denumerator: {denumerator}')
        assignment = numerator / numerator#torch.sum(numerator, dim=1, keepdim=True)
        if (assignment!=assignment).any(): raise ValueError(f'nan found at assignment: {assignment}')
        return assignment# numerator / torch.sum(numerator, dim=1, keepdim=True)
    
    def weights(self, labels: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        '''
        Compute the sample weights according to the cluster and sample labels.
        
        :param labels: FloatTensor of size [batch size]
        :param idx: FloatTensor of size [batch size]
        :return: FloatTensor [batch size, number of clusters]
        '''
        
        if self.cluster_positive_ratio is None:
            return None
        
        _, indices = self.cluster_predicted[:,0].sort()
        predictions = self.cluster_predicted[:,1][indices][idx]
        ratios = self.cluster_positive_ratio[predictions.long()]
        positive_clusters = ratios > self.positive_ratio_threshold
        # compute the weights according to the labels.
        weights = (0.5*(labels==0).float() + (labels==1).float())
        
        # print('\n\n\n')
        # print(f'self.cluster_predicted.device = {self.cluster_predicted.device}')
        # print(f'self.cluster_positive_ratio.device = {self.cluster_positive_ratio.device}')
        # print(f'self.positive_ratio_threshold.device = {self.positive_ratio_threshold.device}')
        # print(f'predictions.device = {predictions.device}')
        # print(f'ratios.device = {ratios.device}')
        # print(f'positive_clusters.device = {positive_clusters.device}')
        # print(f'weights.device = {weights.device}')
        # print('\n\n\n')
        
        # only apply to positive ratio clusters.
        weights *= positive_clusters.float()
        # set "weight" to 1 for all other clusters
        weights += (positive_clusters==0).float()
        return weights
        
    def __setattr__(self, name, value, *args, **kwargs):
        if name in ['cluster_predicted', 'cluster_positive_ratio',]:
            # print(f'setting attribute {name} : {type(value)}')
            if value is None:
                self.__dict__[name] = value
            else:
                self.__dict__[name] = Parameter(
                    torch.tensor(value, dtype = float,)
                ).to(next(self.parameters()).device)
        else:
            super().__setattr__(name, value)
"""
    def samples_distribution(self, batch: Tuple[torch.Tensor], PotentialScores: torch.Tensor) -> torch.Tensor:
        '''
        TODO:
        
        :param idx: FloatTensor of size [batch size]
        :param bounding_boxes: FloatTensor of size [batch size]
        :return: FloatTensor [batch size, number of clusters]
        '''
        _, labels, idxs, boxs, videos, frames = batch
        video_list = []
        label_list = []
        for C in range(self.cluster_number):
            C_bool = self.cluster_predicted[:,1]==C
            C_inds = self.cluster_predicted[:,0][C_bool]
            video_list.append(videos[C_inds])
            label_list.append(labels[C_inds])
            video_frames = videos[C_inds]*10000 + frames[C_inds]
            dsd = DSD(boxs[C_inds], video_frames)
            dsd_idx = idxs[C_inds][dsd]
            DCD_count[C] = len(dsd)
"""       
        
        
        
        
        
        
        
        




























