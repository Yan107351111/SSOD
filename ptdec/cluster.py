import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional


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
        super(WeightedClusterAssignment, self).__init__()
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
        
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power * self.weights(labels, idx)
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
    
    def weights(self, labels: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        '''
        :param labels: FloatTensor of size [batch size]
        :param idx: FloatTensor of size [batch size]
        :return: FloatTensor [batch size, number of clusters]
        '''
        if self.cluster_positive_ratio is None:
            return None
        # compute the weights according to the labels.
        weights = (0.5*(labels==0).float() + (labels==1).float())
        # only apply to positive ratio clusters.
        weights *= (self.cluster_positive_ratio[self.cluster_predicted[idx]])
        # set "weight" to 1 for all other clusters
        weights += (self.cluster_positive_ratio[self.cluster_predicted[idx]]==0).int()
        return weights

        
        
        
        




























