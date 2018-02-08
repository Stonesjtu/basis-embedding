import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

from basis_module import BasisModule

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

    def __init__(self, in_features, out_features, num_basis, num_clusters, use_bias=True, preload_weight_path=None):
        super(BasisLinear, self).__init__(out_features, in_features, num_basis, num_clusters)

        # get integer in python3
        self.features_per_basis = in_features // num_basis
        self.bias = Parameter(0.01 * torch.randn(out_features))

    def enable_basis(self):
        super(BasisLinear, self).enable_basis()
        aux_codebook = []
        for nb, cur_basis_coord in enumerate(self.pq.codebook.t()):
            aux_codebook.append(cur_basis_coord + nb*self.num_clusters)
        self.aux_codebook = Variable(torch.cat(aux_codebook).contiguous().cuda())

    def forward(self, input):
        if self.basis:
            inputs = input.contiguous().view(-1, self.num_sub, self.features_per_basis) # N X Nb X in_ft/Nb
            inputs = inputs.transpose(0, 1).transpose(1, 2) # Nb X in_ft/Nb X N
            output = torch.bmm(self.pq.centroid, inputs) # Nb X Nc X N
            # with torch.autograd.profiler.profile() as prof:
            output = self._decode(output).view(input.size(0), input.size(1), -1)
            # print(prof)
        else:
            output = F.linear(input, self.original_matrix, self.bias)

        return output

    def _decode(self, output):
        """Decode the likelihood of per basis and per clusters into per word"""
        # output = output.transpose(0, 1) # Nc X Nb X N
        output = output.view(-1, output.size(2))
        #TODO: optimize this time consuming part(90% of output layer)
        # coordinates = self.pq.codebook.unsqueeze(2).expand(
        #     *self.pq.codebook.size(),
        #     output.size(2),
        # )
        # likelihoods = output.gather(0, Variable(coordinates))
        likelihoods = output.index_select(0, self.aux_codebook).view(
            self.num_samples, self.num_sub, output.size(-1)
        )
        likelihoods = likelihoods.sum(dim=1).t() # N X V
        likelihoods = likelihoods + self.bias # auto broadcasting
        return likelihoods
