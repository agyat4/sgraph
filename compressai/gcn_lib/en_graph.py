import math
import numpy as np
import torch
import time

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features_v=128, out_features_v=128, in_features_e=128, out_features_e=128, bias=True, node_layer=True):
        super(GraphConvolution, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v

        if node_layer:
            print("this is a node layer")
            self.node_layer = True
            self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            print("this is an edge layer")
            self.node_layer = False
            self.weight = Parameter(torch.FloatTensor(in_features_e, out_features_e))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        device = self.weight.device  # Ensure device consistency
        H_v = H_v.to(device)
        H_e = H_e.to(device)
        adj_e = adj_e.to(device)
        adj_v = adj_v.to(device)
        T = T.to(device)
        batch_size = H_v.shape[0]



        if self.node_layer:
            edge_weights = torch.bmm(H_e, self.p.t().expand(batch_size, -1, -1))  # [batch_size, num_edges, 1]
            T_dense = T.to_dense()
            multiplier1= torch.bmm(
                torch.bmm(T.transpose(1, 2), torch.diag_embed(edge_weights.squeeze(-1))),
                T_dense
            )
            
            mask = torch.eye(multiplier1.size(1), device=multiplier1.device).unsqueeze(0).expand(batch_size, -1, -1)
            M = mask + (1 - mask) * multiplier1
            
            adjusted_A = M * adj_v.to_dense()
            output = torch.bmm(adjusted_A, torch.bmm(H_v, self.weight.unsqueeze(0).expand(batch_size, -1, -1)))
            
            if self.bias is not None:
                output = output + self.bias
             
            return output, H_e

        else:
            node_weights = torch.bmm(H_v, self.p.t().expand(batch_size, -1, -1)) 
            T_dense = T.to_dense()
            multiplier2 = torch.bmm(
                torch.bmm(T_dense, torch.diag_embed(node_weights.squeeze(-1))),
                T_dense.transpose(-2, -1)
            )
            
            mask = torch.eye(multiplier2.size(1), device=multiplier2.device).unsqueeze(0).expand(batch_size, -1, -1)
            M3 = mask + (1 - mask) * multiplier2
            
            adjusted_A = M3 * adj_e.to_dense()
            normalized_adjusted_A = adjusted_A / adjusted_A.max(dim=1, keepdim=True)[0]
            
            output = torch.bmm(normalized_adjusted_A, torch.bmm(H_e, self.weight.unsqueeze(0).expand(batch_size, -1, -1)))
            
            if self.bias is not None:
                output = output + self.bias
           
            return H_v, output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features_v) + ' -> ' \
               + str(self.out_features_v) + ')'