import torch
from torch.nn import Parameter
import torch.nn.functional as F

from .basis_module import BasisModule


class BasisLinear(BasisModule):
    """A module to partially decode every basis in embedding chunks

    Args:
        - in_features: size of each input sample
        - out_features: size of vocabulary
        - num_clusters: number of clusters per basis
        - bias: same as nn.Linear, whether to use bias
        - num_basis: number of chunks that should be normalized at parallel

    Shape:
        - Input: (N, in\_features)
        - intermediate Output: (Nb, N, num_clusters)
        - Output: (N, V)

    Attributes:
        - bias: trainable bias in shape (num_clusters)
    """

    def __init__(self, in_features, out_features, num_basis, num_clusters, bias=True, preload_weight_path=None):
        super(BasisLinear, self).__init__(out_features, in_features, num_basis, num_clusters)

        # get integer in python3
        self.features_per_basis = in_features // num_basis
        self.use_bias = bias
        if bias:
            self.bias = Parameter(0.01 * torch.randn(out_features))
        else:
            self.bias = None

    def forward(self, input):
        input = input.contiguous()
        if self.basis:
            inputs = input.view(-1, self.num_sub, self.features_per_basis)  # N X Nb X in_ft/Nb
            inputs = inputs.transpose(0, 1).transpose(1, 2)  # Nb X N X in_ft/Nb
            output = torch.bmm(self.pq.centroid, inputs)
            output = self._decode(output).view(input.size(0), input.size(1), -1)
        else:
            output = F.linear(input, self.weight, self.bias)

        return output

    def _decode(self, output):
        """Decode the likelihood of per basis and per clusters into per word"""
        output = output.transpose(0, 1)  # Nc X Nb X N
        # TODO: optimize this time consuming part(90% of output layer)
        coordinates = self.pq.codebook.unsqueeze(2).expand(
            *self.pq.codebook.size(),
            output.size(2),
        )
        likelihoods = output.gather(0, coordinates)
        likelihoods = likelihoods.sum(dim=1).t()  # N X V
        if self.use_bias:
            likelihoods = likelihoods + self.bias  # auto broadcasting
        return likelihoods
