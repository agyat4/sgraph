import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_local import window_partition, window_reverse
from .torch_edge_sparse import SparseKnnGraph
from .graph_conv import CustomTransfConv


# Utility functions for flattening and unflattening nodes
def flat_nodes(x, shape):
    B, C, W, H = shape
    x = x.reshape((-1, C, H * W))  # Flatten the spatial dimensions
    x = x.transpose(1, 2)  # Transpose for graph operation
    x = x.reshape((B * H * W, C))  # Reshape into a flattened form
    return x

def unflat_nodes(x, shape):
    B, C, H, W = shape
    x = x.reshape((B, H * W, C))  # Reshape back into spatial form
    x = x.transpose(1, 2)  # Transpose back
    x = x.reshape((-1, C, H, W))  # Final reshaping into the original image form
    return x





class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        avg_out = self.shared_mlp(avg_pool.view(avg_pool.size(0), -1))
        max_out = self.shared_mlp(max_pool.view(max_pool.size(0), -1))
        out = torch.sigmoid(avg_out + max_out).view(x.size(0), -1, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = torch.sigmoid(self.conv(concat))
        return x * attention_map

# CBAM (Convolutional Block Attention Module) implementation
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# WindowGrapherPyg (Graph Convolution) module
class WindowGrapherPyg(nn.Module):
    def __init__(self, dim, window_size, knn=9, conv='transf_custom', heads=8, use_edge_attr=False, dissimilarity=False):
        super(WindowGrapherPyg, self).__init__()
        self.channels = dim
        self.n = window_size * window_size  # number of nodes
        self.window_size = window_size
        self.use_edge_attr = use_edge_attr
        self.knn = knn
        self.heads = heads
        self.dissimilarity = dissimilarity
        self.CustomKnn = SparseKnnGraph(k=self.knn, dissimilarity=self.dissimilarity, loop=False)

        if(conv == 'transf_custom'):
            if(self.use_edge_attr):
                self.conv = CustomTransfConv(dim=dim, heads=heads, edge_dim=1, flow='source_to_target')
            else:
                self.conv = CustomTransfConv(dim=dim, heads=heads, flow='source_to_target')
        else:
            raise NotImplementedError(f'Graph conv {conv} not implemented')

        self.linear_heads = nn.Identity()
        self.output_dim = dim

    def create_custom_graph(self, x):
        edge_index = self.CustomKnn(x)
        return edge_index

    def get_edge_attribute(self, x, edge_index, shape):
        def _get_distances_matrix(H, W):
            coords_h = torch.arange(H, dtype=torch.float)
            coords_w = torch.arange(W, dtype=torch.float)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            return torch.pow(relative_coords.sum(-1), 2)  # Wh*Ww, Wh*Ww

        B, C, H, W = shape
        if(not self.use_edge_attr):
            return None
        relative_pos = _get_distances_matrix(H, W).to(device=x.device)

        row, col = edge_index
        edge_attr = relative_pos[col % (H * W), row % (H * W)].unsqueeze(-1)
        return edge_attr

    def forward(self, x):
        B, C, H, W = x.shape
        if W % self.window_size != 0:
            x = F.pad(x, (0, self.window_size - W % self.window_size))
        if H % self.window_size != 0:
            x = F.pad(x, (0, 0, 0, self.window_size - H % self.window_size))

        _, _, pH, pW = x.shape
        x = window_partition(x, window_size=self.window_size)
        wB, wC, wH, wW = x.shape

        edge_index = self.create_custom_graph(x)
        x = flat_nodes(x, x.shape)

        edge_attr = self.get_edge_attribute(x, edge_index, shape=(wB, wC, wH, wW))
        if(edge_attr is not None):
            x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            x = self.conv(x=x, edge_index=edge_index)

        x = self.linear_heads(x)
        x = unflat_nodes(x, (wB, self.output_dim, wH, wW))  # B, C, H, W

        x = window_reverse(x, self.window_size, H=pH, W=pW)
        x = x[:, :, :H, :W]

        return x


# Parallel Branch Network with Window Graph and CBAM
class Win_Graph_Cbam(nn.Module):
    def __init__(self, in_channels, window_size, knn, conv, heads, use_edge_attr,dissimilarity, reduction=16):
        super(Win_Graph_Cbam, self).__init__()
        # Graph Convolution (WindowGrapherPyg)
        self.graph_branch = WindowGrapherPyg(
            dim=in_channels,
            window_size=window_size,
            knn=knn,
            conv='transf_custom',
            heads=heads,
            use_edge_attr=True,
            dissimilarity=False
        )

        # CBAM (Attention-based) Branch
        self.cbam_branch = CBAM(in_channels, reduction)

        # Fusion Layer (Can use addition or concatenation)
        self.fusion = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Apply both branches in parallel
        graph_features = self.graph_branch(x)
        cbam_features = self.graph_branch(self.cbam_branch(x))

        # Concatenate the features (could also try element-wise addition)
        fused_features = torch.cat([graph_features, cbam_features], dim=1)

        # Fuse with a 1x1 convolution to combine the features
        fused_output = self.fusion(fused_features)

        return fused_output


# Example Usage
if __name__ == '__main__':
    device = "cuda"
    x = torch.rand((2, 192, 64, 64)).to(device)

    model = Win_Graph_Cbam(
        in_channels=192,
        window_size=8,
        knn=9,
        heads=8,
        reduction=16
    ).to(device)

    print(model(x).shape)  # Expected output shape: (2, 192, 64, 64)
