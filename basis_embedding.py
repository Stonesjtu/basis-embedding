# Author: Kaiyu Shi

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable

from utils import basis_cluster

class BasisEmbedding(nn.Module):
    """A class to use basis decomposition to reduce parameters

    Ref: LightRNN (nips2016)
    Arguments:
        - ntoken: vocabulary size
        - emsize: embedding size
        - num_basis: number of basis
        - num_clusters: the number of clusters in each base
        - preload_weight_path: the path to load a pre-trained weight matrix

    Attributes:
        - coordinates: (V, Nb) the coordinates of words under specific basis
        - weight: (Nb, Nc, E/Nb)the cluster centroids of original embedding matrix

    Shape:
        - Input: (B, N) indices of words
        - Output: (B, N, embedding_dim)
    """

    def __init__(self, ntoken, emsize, num_basis=2, num_clusters=400, preload_weight_path=None):
        super(BasisEmbedding, self).__init__()
        self.ntoken = ntoken
        self.emsize = emsize
        self.num_basis = num_basis
        self.num_clusters = num_clusters
        if not emsize % num_basis == 0:
            raise ValueError('Embedding size({}) should be '
                             'divisible by basis number({})'.format(emsize, num_basis))
        # self.coordinates = torch.zeros(ntoken, num_basis)
        pre_weight = torch.zeros(ntoken, emsize).uniform_(-0.1, 0.1)
        if preload_weight_path is not None:
            emb = nn.Embedding(ntoken, emsize)
            emb.load_state_dict(torch.load(preload_weight_path))
            pre_weight = emb.weight.data

        basis, coordinates = basis_cluster(pre_weight, num_basis, num_clusters)
        self.weight = Parameter(basis)
        # self.weight = Parameter(0.01 * torch.randn(basis.size()))
        self.register_buffer('coordinates', Variable(coordinates))


    def forward(self, input):
        # TODO: add padding indice
        coordinates = self.coordinates[input.contiguous().view(-1)].t() # Nb X N
        partial_embeddings = []
        for cur_basis in range(self.num_basis):
            partial_embedding = self.weight[cur_basis][coordinates[cur_basis]] # N X E/Nb
            partial_embeddings.append(partial_embedding)

        embeddings = torch.cat(partial_embeddings, dim=1)
        return embeddings.view(input.size(0), input.size(1), -1)
