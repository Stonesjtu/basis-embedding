import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

from utils import basis_cluster

class ProductQuantizer(nn.Module):
    """Product Quantizer for pytorch

    The API simply follows the ProductQuantizer in `faiss` project

    Parameters:
        - dimension: dimensionality of the input vectors
        - num_sub: L number of sub-quantizers (M)
        - k: number of clusters per sub-vector index (2^^nbits)

    Attributes:
        - codebook: (V, Nb) the coordinates of words under specific basis
        - centroid: (Nb, Nc, E/Nb)the cluster centroids of original embedding matrix


    """

    def __init__(self, dimension, num_sub, k):
        super(ProductQuantizer, self).__init__()
        self.dimension = dimension
        self.num_sub = num_sub
        self.k = k
        if not dimension % num_sub == 0:
            raise ValueError('Embedding size({}) should be '
                             'divisible by basis number({})'.format(dimension, num_sub))

    def train_code(self, data_matrix):
        """Get the codebook and centroids from data

        Args:
            - data_matrix: (N, D) where N is number of vectors
            D is dimension

        Returns:
            None
        """

        centroid, codebook = basis_cluster(data_matrix.cpu(), self.num_sub, self.k)
        self.centroid = Parameter(centroid)
        self.register_buffer('codebook', codebook)
        if data_matrix.is_cuda:
            self.cuda()

    def get_centroid(self, index=None):
        """Get the reproduction value for training data

        Args:
            index: `Tensor` (C) the index(es) to look-up
        """
        if index is None:
            code = self.codebook
        else:
            code = self.codebook[index]
        return self.decode(Variable(code))

    def decode(self, code):
        """Decode the code into reproduction value

        The reproduction value is a concatenation of centroids in
        different sub-quantizer.

        Args:
            - code: (C, N_s) where C is arbitrary number of codes
            N_s is number of sub-quantizers

        Return:
            - centroid: (C, D) the reproduction values of input codes
        """
        sub_centroids = []
        for cur_sub in range(self.num_sub):
            cur_index = code[:, cur_sub]
            sub_centroid = self.centroid[cur_sub][cur_index] # N X E/Nb
            sub_centroids.append(sub_centroid)

        centroid = torch.cat(sub_centroids, dim=1)
        return centroid


    def compute_code(self, point):
        pass
