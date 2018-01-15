# the NCE module written for pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

from basis_linear_softmax import BasisLinear

class BasisLoss(nn.Module):
    """Cross Entropy loss for basis decomposition decoder

    Args:
        nhidden: hidden size of LSTM(a.k.a the output size)
        num_clusters: number of clusters (compared with output size)
        num_basis: number of basis (should be compatible with coordinates)
        size_average: average the loss by batch size

    Shape:
        - decoder: :math:`(E, V)` where `E = embedding size`
        - input: (N, nhidden)
        - out: a scalar loss Variable

    """

    def __init__(self,
                 nhidden,
                 num_clusters,
                 num_basis,
                 ntokens,
                 size_average=True,
                 preload_weight_path=None,
                 ):
        super(BasisLoss, self).__init__()

        self.nhidden = nhidden
        self.num_clusters = num_clusters
        self.num_basis = num_basis

        self.decoder = BasisLinear(
            nhidden, ntokens, num_basis, num_clusters,
            preload_weight_path=preload_weight_path,
        )
        self.criterion = nn.CrossEntropyLoss(size_average=size_average)

    def forward(self, input, target):
        """compute the loss with output and the desired target

        Parameters:
            input: the output of the RNN model, being an predicted embedding
            target: the supervised training label.

        Shape:
            - input: :math:`(N, E)` where `N = number of tokens, E = embedding size`
            - target: :math:`(N)`

        Return:
            the scalar BasisLoss Variable ready for backward
        """

        decoded = self.decoder(input).contiguous()
        loss = self.criterion(decoded,target)
        return loss
