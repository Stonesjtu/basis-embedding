# Some utilities functions to help abstract the codes
# Author: Kaiyu Shi
# Time: Wed 09 Aug 2017 10:46:25 AM CST

import numpy as np

import torch

def basis_cluster(weight, num_basis, num_clusters, cuda=False):
    """Divide the weight into `num_basis` basis and clustering

    Params:
        - weight: weight matrix to do basis clustering
        - num_basis: number of basis, also the dimension of coordinates
        - num_cluster: number of clusters per basis

    Return:
        - basis: (Nb, Nc, E/Nb)the cluster centers for each basis.
        - coordinates: (V, Nb) the belongings for basis of each token.
    """
    partial_embeddings = weight.chunk(num_basis, dim=1)

    coordinates = []
    basis = []
    if not cuda:
        from sklearn.cluster import KMeans
        clustor = KMeans(init='k-means++', n_clusters=num_clusters, n_init=1)
    for partial_embedding in partial_embeddings:
        if cuda:
            from libKMCUDA import kmeans_cuda
            centroid, coordinate = kmeans_cuda(partial_embedding.numpy(), num_clusters, seed=7)
            # some clusters may have zero elements, thus the centroids becomes [nan] in libKMCUDA
            centroid = np.nan_to_num(centroid)
        else:
            clustor.fit(partial_embedding.numpy())
            centroid, coordinate = clustor.cluster_centers_, clustor.labels_
        basis.append(torch.from_numpy(centroid.astype('float')))
        coordinates.append(torch.from_numpy(coordinate.astype('int32')))

    basis = torch.stack(basis).float() # Nb X Nc(clusters) X E/Nb
    coordinates = torch.stack(coordinates).t().long() # V X Nb(number of basis)
    return basis, coordinates

def get_similarity_count(source, target):
    """Get the similarity counts between source vector and target vectors in matrix"""
    similarity_mat = source == target
    similarity_count = similarity_mat.sum(dim=1)
    return similarity_count

def get_similarity_topk(source, target, topk=10):
    count = get_similarity_count(source, target)
    val, idx = count.topk(dim=0, k=topk, sorted=True)
    return val, idx
