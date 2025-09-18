import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor, OptPairTensor
from torch_geometric.utils import softmax

from torch_geometric.nn import knn_graph
import time
import torch.nn as nn
import sys
    

class CustomTransfConv(MessagePassing):
    def __init__(
        self,
        dim: int,
        heads: int = 1,
        edge_dim: Optional[int] = None,
        **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        print(f'Using custom graph transf w/ {heads} heads')

        self.dim = dim
        self.heads = heads
        self.edge_dim = edge_dim

        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.lin_edge = None
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, dim, bias=False)

        print(f'Using edge dim: {self.lin_edge is not None}\n')
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.qkv.reset_parameters()
        self.proj.reset_parameters()
        
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        

    
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):

        N,C = x.shape
        H = self.heads

        qkv = self.qkv(x).reshape(N,3,H,C//H).permute(1,0,2,3)

        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        out = self.propagate(edge_index, query=q, key=k, value=v,
                             edge_attr=edge_attr, size=None)
        
   
        out = out.view(-1, self.dim) # concat
        
        out = self.proj(out)

        return out + x
        
    
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        
        # query_i.shape         N,H,C 

        att = query_i * key_j #/ math.sqrt(self.dim)
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,self.head_dim)
            att = att + edge_attr
        att = att.sum(dim=-1)
        alpha = softmax(att, index, ptr, size_i)

        out = value_j * alpha.view(-1, self.heads, 1)
        return out
    
