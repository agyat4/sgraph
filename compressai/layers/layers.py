# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import torch
import torch.nn as nn
import torch.nn.functional as F

from .win_attention import WinBasedAttention
from .window_cbam import WindowCBAM
from ..gcn_lib.local_graph_pyg import WindowGrapherPyg
from ..gcn_lib.cbam_graph import Win_Graph_Cbam
from ..gcn_lib.en_graph import  GraphConvolution

__all__ = [
    "conv3x3",
    "subpel_conv3x3",
    "conv1x1",
    "Win_noShift_Attention",
    "Win_GraphPyg",
    "Spectral_Graph"

]


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualUnit(nn.Module):
    """Simple residual unit."""

    def __init__(self, dim):

        super().__init__()
        self.conv = nn.Sequential(
            conv1x1(dim, dim // 2),
            nn.GELU(),
            conv3x3(dim // 2, dim // 2),
            nn.GELU(),
            conv1x1(dim // 2, dim),
        )
        self.relu = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        out = self.relu(out)
        return out



class Win_noShift_Attention(nn.Module):
    """Window-based self-attention module."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
        super().__init__()
        N = dim

        '''class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.GELU(),
                    conv3x3(N // 2, N // 2),
                    nn.GELU(),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.GELU()

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out'''

        self.conv_a = nn.Sequential(ResidualUnit(N), ResidualUnit(N), ResidualUnit(N))

        self.conv_b = nn.Sequential(
            WindowCBAM(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size),
            ResidualUnit(N),
            ResidualUnit(N),
            ResidualUnit(N),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

class Win_GraphPyg(nn.Module):
    """Window-based graph pyg module."""

    def __init__(self, dim, window_size=8, knn = 9, conv = 'transf', heads = 8, use_edge_attr = False, dissimilarity = False):
        super().__init__()
        N = dim

        self.conv_a = nn.Sequential(ResidualUnit(N), ResidualUnit(N), ResidualUnit(N))

        self.conv_b = nn.Sequential(
            WindowGrapherPyg(dim=dim, window_size=window_size, knn=knn, conv=conv, heads=heads, use_edge_attr=use_edge_attr, dissimilarity=dissimilarity), #  Win_Graph_Cbam(in_channels=dim, window_size=window_size, knn=knn, conv=conv, heads=heads, use_edge_attr=use_edge_attr, dissimilarity=dissimilarity),

            ResidualUnit(N),
            ResidualUnit(N),
            ResidualUnit(N),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out



class Spectral_Graph(nn.Module):

    def __init__(self, nfeat_v=6, nfeat_e=6, nhid=16, nclass=128, dropout=0.5, node_layer=True):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e, node_layer=False)
        self.gc3 = GraphConvolution(nhid, nclass, nfeat_e, nfeat_e, node_layer=True)
        self.dropout = dropout

    def forward(self, X, Z, adj_e, adj_v, T, pooling=1, node_count=1, graph_level=True):
        num_edges= Z.shape[1]
        num_nodes= X.shape[1]
        '''print("shape of edge", Z.shape)
        print("shape of node", X.shape)'''

        X = torch.permute(X, (0, 3, 1, 2))
        X = X.reshape(-1, num_nodes, 128)  # Flatten batch and graph dimensions
        Z = torch.permute(Z, (0, 3, 1, 2))
        Z = Z.reshape(-1,num_edges, 128)
        adj_v = adj_v.reshape(-1, num_nodes, num_nodes)
        adj_e = adj_e.reshape(-1, num_edges, num_edges)
        T = T.reshape(-1, num_edges, num_nodes)

        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc1[0]), F.relu(gc1[1])
       

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)
        
        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc2[0]), F.relu(gc2[1])
      
        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc3(X, Z, adj_e, adj_v,T)

        #return F.log_softmax(X, dim=1)'''
        X = X.view(-1, 16, num_nodes, X.size(-1))  # Assuming 3 graphs per batch
        X = torch.permute(X, (0, 2, 3, 1))

        # Restore original shape for Z
        Z = Z.view(-1, 16, num_edges, Z.size(-1))  # Assuming 3 graphs per batch
        Z = torch.permute(Z, (0, 2, 3, 1))
       
        return X, Z
        #return F.log_softmax(X, dim=2), F.log_softmax(Z, dim=2)
