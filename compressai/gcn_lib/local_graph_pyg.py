import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .torch_local import window_partition, window_reverse
from .torch_edge_sparse import SparseKnnGraph
from .graph_conv import CustomTransfConv

def flat_nodes(x,shape):
    B,C,W,H = shape
    x = x.reshape((-1,C,H*W))#.contiguous()
    x = x.transpose(1,2)# .contiguous()
    x = x.reshape((B*H*W,C))# .contiguous()
    return x

def unflat_nodes(x,shape):
    B,C,H,W = shape

    x = x.reshape((B,H*W,C))
    x = x.transpose(1,2)
    x = x.reshape((-1,C,H,W))
    return x

class WindowGrapherPyg(nn.Module):
    """
    Window Grapher module with graph convolution in pytorch geometric
    """
    def __init__(self, dim, window_size, knn=9, conv='transf_custom', heads = 8, use_edge_attr = False, dissimilarity = False):
        super(WindowGrapherPyg, self).__init__()
        self.channels = dim
        self.n = window_size * window_size # number of nodes
        self.window_size = window_size
        self.use_edge_attr = use_edge_attr
        self.knn = knn
        self.heads = heads
        self.dissimilarity = dissimilarity
        self.CustomKnn = SparseKnnGraph(k=self.knn,dissimilarity=self.dissimilarity, loop=False)

        if(conv == 'transf_custom'):
            if(self.use_edge_attr):
                self.conv = CustomTransfConv(dim= dim, heads= heads, edge_dim=1, flow = 'source_to_target')
            else:
                self.conv = CustomTransfConv(dim= dim, heads= heads, flow = 'source_to_target')
        else:
            raise NotImplementedError(f'Graph conv {conv} not implemented in gcn_lib.local_graph_pyg.py')
        
        
        self.linear_heads = nn.Identity()

        # if(conv == 'transf'):
        #     self.linear_heads = nn.Linear(in_features = dim, out_features = dim, bias = True)
            
        self.output_dim = dim

    
    def create_custom_graph(self,x):
        edge_index = self.CustomKnn(x)
        return edge_index
            
    def get_edge_attribute(self, x, edge_index, shape):
        def _get_distances_matrix(H,W):
            coords_h = torch.arange(H, dtype=torch.float)
            coords_w = torch.arange(W, dtype=torch.float)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            return torch.pow(relative_coords.sum(-1) ,2) # Wh*Ww, Wh*Ww
        
        B, C, H, W = shape

        if(not self.use_edge_attr):
            return None
        
        relative_pos = _get_distances_matrix(H,W).to(device=x.device)

        row, col = edge_index
        edge_attr = relative_pos[col%(H*W), row%(H*W)].unsqueeze(-1)
        return edge_attr



    def forward(self, x):
       
        B, C, H, W = x.shape
        
        if W % self.window_size != 0:
            x = F.pad(x, (0, self.window_size - W % self.window_size))
        if H % self.window_size != 0:
            x = F.pad(x, (0, 0, 0, self.window_size - H % self.window_size))
        
        _,_, pH, pW = x.shape
        x = window_partition(x, window_size= self.window_size)
        wB, wC, wH, wW = x.shape
        
        if(self.knn is not None and self.knn > 0):
            edge_index = self.create_custom_graph(x)

        else:
            raise NotImplementedError(f'Cannot use knn = 0 or None on the entire tensor..')
    
        x = flat_nodes(x, x.shape)

        edge_attr = self.get_edge_attribute(x, edge_index, shape=(wB, wC, wH, wW))
        if(edge_attr is not None):
            x = self.conv(x = x, edge_index = edge_index, edge_attr = edge_attr)
        else:
            x = self.conv(x = x, edge_index = edge_index)

        x = self.linear_heads(x)
        x = unflat_nodes(x, (wB,self.output_dim,wH,wW)) # B, C, H, W

        x = window_reverse(x, self.window_size, H=pH, W=pW)
        x = x[:,:,:H,:W]
        
        return x



if __name__ == '__main__':

    device = "cuda"
    x = torch.rand((2,192,64,64)).to(device)

    l = WindowGrapherPyg(
        dim=192,
        window_size=8,
        knn=9,
        conv='transf_custom',
        heads=8,
        use_edge_attr=True,
        dissimilarity=False
    ).to(device)
    

    print(l(x).shape)
