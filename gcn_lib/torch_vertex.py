# 2024.07.26 -- Changed for building SViG 

# Adapted from https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, act_layer
from .torch_edge import DenseDilatedKnnGraph
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch_scatter import scatter

class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        B, C, N, _ = x.shape
        x_all_batches = x.squeeze(-1) ## B, C, N
        x_all_batches = x_all_batches.permute(1, 0, 2) ## C, B, N
        x_all_batches = x_all_batches.reshape(C, -1, 1).squeeze(-1) ## C, N*B 
        x_all_batches = x_all_batches.permute(1,0) ##    N*B ,C
        x_i = x_all_batches[edge_index[0]] #Destination nodes
        x_j = x_all_batches[edge_index[1]] #Source nodes (neighborhood)
        x_j = x_j - x_i ## number_of_edges, C
        #Torch scatter doesn't support half operations
        out = scatter(src=x_j.to(torch.float32), index=edge_index[0], dim=0, dim_size=x_all_batches.size(0), reduce='max') ## N(total), C
        x_j = out.to(torch.float16).unsqueeze(-1).reshape(B, N, C) # B, N, C
        x_j = x_j.permute(0,2,1).unsqueeze(3) # B, C, N, 1
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, threshold=0.9, conv='edge', act='relu',
                 norm=None, bias=True):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.dilated_knn_graph = DenseDilatedKnnGraph(threshold)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x)
        x = super(DyGraphConv2d, self).forward(x, edge_index) #The actual graph convolution    
        return x.reshape(B, -1, H, W).contiguous()

class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, threshold=9, conv='edge', act='relu', norm=None,
                 bias=True, n=196, drop_path=0.0):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, threshold, conv,
                              act, norm, bias)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        # B is the batch size, C is the number of channels, H is the height and W is the width
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
    