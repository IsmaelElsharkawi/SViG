# 2024.07.26 -- Changed for building SViG 

# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
import cupyx 
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def select_neighbors(distance, distribution_theshold):
    """Selects the distribution of the distances that will be used to get the neighbors
    Args:
        distance: (batch_size, num_points, num_points)
    Returns:
        Adjacency Matrix of True and False (i.e. which edges to take): (batch_size, num_points, num_points)
    """
    similarity_matrix = -distance
    # change the thresholds s.t. it's for each node it's probaly the case that if we take
    # avg on dimension 2, then the target is going to be in dimension 1, and the source is in dimension 2 (for the edges)
    similarity_mean = torch.mean(similarity_matrix, (1,2), keepdim = True)
    similarity_std = torch.std(similarity_matrix, (1,2), keepdim = True)
    similarity_matrix = (similarity_matrix - similarity_mean)/similarity_std
    threshold = norm.ppf(distribution_theshold)

    if(distribution_theshold<=0):
        print("Threshold is: ", distribution_theshold, " >> resorting to a fully connected graph :((" )
        threshold = float('-inf')
    adjacency_matrix = (similarity_matrix > threshold)
    return adjacency_matrix

def get_edge_indices(adj_matrices, n_points):
    # Get the size of each adjacency matrix
    # size = adj_matrices[0].shape[0]

    # # Create an empty tensor to hold the block-diagonal matrix
    # sparse_adj_matrix = torch.zeros(size * len(adj_matrices), size * len(adj_matrices))

    # # Fill the block-diagonal matrix with the adjacency matrices
    # for i, adj_matrix in enumerate(adj_matrices):
    #     sparse_adj_matrix[i*size : (i+1)*size, i*size : (i+1)*size] = adj_matrix
    # edge_index = torch.nonzero(sparse_adj_matrix, as_tuple=False).t().to(adj_matrices.device)  
    # del sparse_adj_matrix
    ######################## EFFICIENT IMPLEMENTATION
    temp = torch.nonzero(adj_matrices, as_tuple=False)
    temp[:,0] = temp[:,0] * n_points 
    temp[:,1] = temp[:,0] + temp[:,1] 
    temp[:,2] = temp[:,0] + temp[:,2] 
    edge_index = temp[:,1:].t()
    if not adj_matrices.bool().any():
        print(adj_matrices)
        print(edge_index)
        # Something went terribly wrong
    return edge_index

def dense_knn_matrix(x, k):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int, represents the threshold or the k (in case of an error)
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = pairwise_distance(x.detach())
        adjacency_matrix = select_neighbors(dist, k)
        adjacency_matrix = adjacency_matrix.to(x.device)

        if adjacency_matrix is not None:
            ############## NEW PART ################
            edge_indices = get_edge_indices(adjacency_matrix, n_points)
            return edge_indices
        else:
            #resotring to k-NN in case the adjacency matrix is not initialized correctly
            #Something must have gone terribly wrong if this branch is taken in any layer
            _, nn_idx = torch.topk(-dist, k=9) 
            max_k = k
            center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, max_k, 1).transpose(2, 1)
            return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9):
        super(DenseDilatedKnnGraph, self).__init__()
        self.k = k
        
    def forward(self, x):
        #### normalize
        x = F.normalize(x, p=2.0, dim=1)
        ####
        edge_index = dense_knn_matrix(x, self.k)
        return edge_index