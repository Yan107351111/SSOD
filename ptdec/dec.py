import torch
import torch.nn as nn

from ptdec.cluster import ClusterAssignment, WeightedClusterAssignment


class DEC(nn.Module):
    def __init__(self,
                 cluster_number: int,
                 hidden_dimension: int,
                 encoder: torch.nn.Module,
                 alpha: float = 1.0):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(cluster_number, self.hidden_dimension, alpha)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        return self.assignment(self.encoder(batch))


class WDEC(nn.Module):
    def __init__(self,
                 cluster_number: int,
                 hidden_dimension: int,
                 feature_extractor: torch.nn.Module,
                 encoder: torch.nn.Module,
                 positive_ratio_threshold: float = 0.6,
                 alpha: float = 1.0):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param feature_extractor: feature_extractor to use
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(WDEC, self).__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = WeightedClusterAssignment(cluster_number, self.hidden_dimension, positive_ratio_threshold, alpha)

    def forward(self, batch: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :param labels: [batch size,] FloatTensor
        :param idx: [batch size,] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        with torch.no_grad():
            batch = self.feature_extractor(batch)
        return self.assignment(self.encoder(batch), labels, idx)
