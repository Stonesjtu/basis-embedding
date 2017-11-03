# Author: Kaiyu Shi

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable

from utils import basis_cluster

class BasisLinear(nn.Module):
    """A module to partially decode every basis in embedding chunks

    Args:
        - in_features: size of each input sample
        - out_features: size of vocabulary
        - num_clusters: number of clusters per basis
        - bias: same as nn.Linear
        - num_basis: number of chunks that should be normalized at parallel
        - preload_weight_path: the path to load a pre-trained weight matrix

    Shape:
        - Input: (N, in\_features)
        - intermediate Output: (Nb, N, num_clusters)
        - Output: (N, V)

    Attributes:
        weight: chunked weight matrix
        bias: learnable bias in shape (num_clusters)
    """

    def __init__(self, in_features, out_features, num_clusters, num_basis, bias=True, preload_weight_path=None):
        super(BasisLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_clusters = num_clusters
        self.num_basis = num_basis

        if not in_features % num_basis == 0:
            raise ValueError('Input feature size({}) should be '
                             'divisible by basis number({})'.format(in_features, num_basis))

        # get integer in python3
        self.features_per_basis = in_features // num_basis
        pre_weight = 0.01 * torch.randn(out_features, in_features)
        self.bias = Parameter(0.01 * torch.randn(num_basis, num_clusters))
        if preload_weight_path is not None:
            linear = nn.Linear(in_features, out_features)
            linear.load_state_dict(torch.load(preload_weight_path))
            pre_weight = linear.weight.data

        # pre_weight = torch.load('./glove_ptb.pt')
        pre_trained_basis = False
        if pre_trained_basis:
            basis = torch.load('./basis.pt')
            coordinates = torch.load('./coordinates.pt')
        else:
            basis, coordinates = basis_cluster(pre_weight, num_basis, num_clusters)
            # torch.save(coordinates, 'coordinates.pt')
            # torch.save(basis, 'basis.pt')
        self.weight = Parameter(basis)
        # self.weight = Parameter(0.01 * torch.randn(basis.size()))
        self.register_buffer('coordinates', Variable(coordinates.t().contiguous()))

    def forward(self, input):
        inputs = input.view(-1, self.num_basis, self.features_per_basis) # N X Nb X in_ft/Nb
        inputs = inputs.transpose(0, 1).transpose(1, 2) # Nb X N X in_ft/Nb
        output = torch.baddbmm(self.bias.unsqueeze(2), self.weight, inputs)
        output = self._decode(output.transpose(1, 2))
        return output

    def _decode(self, output):
        """Decode the likelihood of per basis and per clusters into per word"""
        output = output.transpose(1, 2) # Nb X Nc X N
        #TODO: optimize this time consuming part(90% of output layer)
        coordinates = self.coordinates.unsqueeze(2).expand(
            self.coordinates.size(0),
            self.coordinates.size(1),
            output.size(2),
        )
        likelihoods = output.gather(1, coordinates)
        likelihoods = likelihoods.sum(dim=0).t() # N X V
        return likelihoods
