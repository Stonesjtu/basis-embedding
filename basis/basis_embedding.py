from .basis_module import BasisModule


class BasisEmbedding(BasisModule):
    """A class to use basis decomposition to reduce parameters

    Ref:
        - LightRNN (nips2016)
        - Product quantization(Jegou)
    Arguments:
        - ntoken: vocabulary size
        - emsize: embedding size
        - num_basis: number of basis
        - num_clusters: the number of clusters in each base

    Shape:
        - Input: (B, N) indices of words
        - Output: (B, N, embedding_dim)
    """

    def __init__(self, ntoken, embedding_dim, num_basis=2, num_clusters=400):
        super(BasisEmbedding, self).__init__(
                ntoken, embedding_dim, num_basis, num_clusters
        )

    def forward(self, input):
        # TODO: add padding indice
        size = input.size()
        input = input.contiguous().view(-1)
        if self.basis:
            embeddings = self.pq.get_centroid(input.data)
        else:
            embeddings = self.weight[input]
        return embeddings.view(*size, -1)

    def get_embedding(self):
        """Get the embeddings for all words"""
        if self.basis:
            return self.pq.get_centroid()
        else:
            return self.weight
