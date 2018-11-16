# the NCE module written for pytorch
import torch
import torch.nn as nn

from basis import BasisLinear
from utils import get_mask

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
                 ntokens,
                 num_basis,
                 num_clusters,
                 size_average=True,
                 preload_weight_path=None,
                 ):
        super(BasisLoss, self).__init__()

        self.nhidden = nhidden
        self.num_basis = num_basis
        self.num_clusters = num_clusters
        self.size_average = size_average

        if num_basis == 0:
            self.decoder = nn.Linear(nhidden, ntokens)
        else:
            self.decoder = BasisLinear(
                nhidden, ntokens, num_basis, num_clusters,
            )
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target, length):
        """compute the loss with output and the desired target

        Parameters:
            input: the output of the RNN model, being an predicted embedding
            target: the supervised training label.

        Shape:
            - input: :math:`(N, E)` where `N = number of tokens, E = embedding size`
            - target: :math:`(N)`

        Return:
            the scalar Variable ready for backward
        """

        decoded = self.decoder(input).contiguous()
        mask = get_mask(length)
        loss = self.criterion(
            decoded.view(-1, decoded.size(2)), target.view(-1)
        ).view(decoded.size(0), decoded.size(1))
        loss = torch.masked_select(loss, mask)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
