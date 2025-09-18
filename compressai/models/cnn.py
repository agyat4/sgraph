import math
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from .utils import conv, deconv, update_registered_buffers
from compressai.ops import ste_round
from compressai.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .base import CompressionModel
from compressai.layers import Win_GraphPyg, Spectral_Graph

import scipy.sparse as sp


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def split_into_patches(feature_map, num_patches):
    B, C, H, W = feature_map.shape

    # Ensure num_patches is a perfect square
    patch_dim = int(math.sqrt(num_patches))
    if patch_dim ** 2 != num_patches:
        raise ValueError("Number of patches must be a perfect square")

    # Calculate patch sizes
    patch_h = H // patch_dim
    patch_w = W // patch_dim

    # Split into local patches
    patches = feature_map.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    patches = patches.contiguous().view(B, C, num_patches, patch_h, patch_w)

    return patches, patch_h
    
def global_avg_pool_patches(patches):
    B, C, N, H, W = patches.shape
    pooled_patches = patches.view(B, C, N, -1).mean(dim=-1)
    return pooled_patches



def expand_to_patches(x, patch_size=8):
    b, g, c, n = x.shape  # b: batch, g: groups, c: channels, n: number of patches
    x = x.contiguous().view(b, g * c, n, 1, 1)
    x = x.expand(b, g * c, n, patch_size, patch_size)  # Expand the single value to patch_size x patch_size
    x = x.permute(0, 1, 2, 3, 4).contiguous()  # Rearrange dimensions
    x = x.view(b, g * c, n, patch_size, patch_size)  # Reshape to final form
    return x

def rearrange_patches_to_full(x):
    b, c, n, h, w = x.shape  # b: batch, c: channels, n: number of patches, h: patch height, w: patch width
    
    # Calculate the side length of the full feature map
    side_length = int(math.sqrt(n) * h)
    
    # Reshape to align patches
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(b, int(math.sqrt(n)), int(math.sqrt(n)), c, h, w)
    
    # Rearrange patches to form full feature
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(b, c, side_length, side_length)
    
    return x


def compute_undirected_edge_features(patches):
    """
    Computes edge features for an undirected cyclic graph.
    Each edge feature is the residual (difference) between the features of two connected nodes.
    
    Args:
        patches: Tensor of shape [B, G, C, N], where:
            B = batch size
            G = number of nodes
            C = feature dimension of each node
            N = number of graphs (patches)
    
    Returns:
        edge_features: Tensor of shape [B, E, C, N], where:
            E = total number of edges (G for cyclic graph)
            C = feature dimension
            N = number of graphs
    """
    B, G, C, N = patches.shape
    edge_features = []
    
    # Create cyclic connections (i to i+1, last to first)
    for i in range(G):
        j = (i + 1) % G  # Ensuring cyclic connection
        edge_feature = patches[:, i, :, :] - patches[:, j, :, :]
        edge_features.append(edge_feature)
    
    edge_features = torch.stack(edge_features, dim=1)  # Shape: [B, G, C, N]
    return edge_features

def create_cyclic_graph(num_nodes):
    """
    Creates a cyclic adjacency matrix where each node is connected to its next neighbor in a closed loop.
    """
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        j = (i + 1) % num_nodes  # Ensure last node connects to first
        adj[i, j] = 1
        adj[j, i] = 1  # Undirected
    
    # Create T matrix
    num_edges = num_nodes  # Since it's a cycle, edges = nodes
    T = torch.zeros(num_nodes, num_edges)
    for i in range(num_nodes):
        T[i, i] = 1
        T[(i + 1) % num_nodes, i] = 1
    
    adj_norm = normalize_adj(adj)
    edge_adj = create_edge_adj(adj)
    
    return adj_norm, T, edge_adj

def normalize_adj(adj):
    adj = adj + torch.eye(adj.shape[0])
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def create_edge_adj(adj):
    num_edges = int(adj.sum().item() // 2)
    edge_adj = torch.eye(num_edges)  # Cyclic graph edges are self-connected
    return edge_adj

def process_data(node_features, edge_features):
    batch_size, num_nodes, seq_len, num_graphs = node_features.shape
    adj_norm, T, edge_adj = create_cyclic_graph(num_nodes)
    
    adj_norm_batch = adj_norm.unsqueeze(0).unsqueeze(1).repeat(batch_size, num_graphs, 1, 1)
    T_batch = T.unsqueeze(0).unsqueeze(1).repeat(batch_size, num_graphs, 1, 1)
    edge_adj_batch = edge_adj.unsqueeze(0).unsqueeze(1).repeat(batch_size, num_graphs, 1, 1)
    
    return node_features, edge_features, adj_norm_batch, T_batch, edge_adj_batch



class PatchEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embedding = nn.Linear(in_features=32, out_features=128)
    
    def forward(self, x):
        B, G, C, N = x.shape
        x = x.permute(0, 1, 3, 2).reshape(-1, C)
        x = self.embedding(x)
        return x.view(B, G, N, -1).permute(0, 1, 3, 2)



class ReversePatchEncoder(nn.Module):
    def __init__(self, in_features=128, out_features=32):
        super().__init__()
        self.rev_embedding = nn.Linear(in_features, out_features)

    def forward(self, x):
        B, G, C, N = x.shape
        x = x.permute(0, 1, 3, 2).reshape(-1, C)
        x = self.rev_embedding(x)
        return x.view(B, G, N, -1).permute(0, 1, 3, 2)

class WACNN(CompressionModel):
    """CNN based model"""

    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(**kwargs)
        self.num_slices = 10
        self.max_support_slices = 5

      
      
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 27, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(27, 27, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(),
        )
        
        self.spatial = nn.Sequential(
            nn.Conv2d(3,6, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.g_s = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),

            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.h_scale_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.scale_conv = nn.ModuleList(
            nn.Sequential(
                conv(32*min(i,5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 320, stride=1,kernel_size=3),
            )for i in range(10)
            )

        self.shift_conv = nn.ModuleList(
                    nn.Sequential(
                        conv(32*min(i,5), 224, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(224, 320, stride=1,kernel_size=3),
                    )for i in range(10)
                    )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 , 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
            )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 , 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
            )

       

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32*min(i+1,1), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


    def forward(self, x):
        
        ########################## spectral branch #############################
        grouped_output = self.grouped_conv(x)
        batch_size, C, H, W = x.shape
        num_patches = 16  # You can change this to 4, 9, 25, 36, etc.

        # Split the features into patches
        patches, patch_dim = split_into_patches(grouped_output, num_patches)
        pooled_patches = global_avg_pool_patches(patches)

        grouped_pooled_patches = pooled_patches.view(batch_size, C, 32, 16)
       
        edge_features = compute_undirected_edge_features(grouped_pooled_patches)
        encoded_node_features = self.patch_encoder(grouped_pooled_patches)
        encoded_edge_features = self.patch_encoder(edge_features)


        nodes, edges, adj, T, edge_adj = process_data(encoded_node_features, encoded_edge_features)

       
        # initialize spectral_graph1 with parameters from grouped_output
     

        o_node, o_edge = self.spectral_graph1(nodes, edges, edge_adj, adj, T)
        
        

        rev_encoded_nodes = self.reverse_patch_encoder(o_node)
        rev_encoded_edges = self.reverse_patch_encoder(o_edge)

        output = torch.cat([rev_encoded_nodes, rev_encoded_edges], dim=1)

        rev_encoded_patches = expand_to_patches(output, patch_dim)
        rev_encoded_patches = rearrange_patches_to_full(rev_encoded_patches)


       # Process g_a with spectral injection
        x = self.g_a[:2](x)  # First conv and GDN
        x = torch.cat([x, rev_encoded_patches], dim=1)
        y = self.g_a[2:](x)  # Rest of g_a layers
    


        ########################## regular branch ##############################
       
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            if not support_slices:
                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            else:
                support_tensor = torch.cat(support_slices, dim=1)
                scale_sft = self.scale_conv[slice_index](support_tensor)
                shift_sft = self.shift_conv[slice_index](support_tensor)
                latent_mean_conditioned = latent_means * (scale_sft + 1) + shift_sft
                latent_scale_conditioned = latent_scales * (scale_sft + 1) + shift_sft
                mean_support =  latent_mean_conditioned #torch.cat([latent_mean_conditioned])+ support_slices, dim=1)
                scale_support = latent_scale_conditioned #torch.cat([latent_scale_conditioned]) + support_slices, dim=1)

           

            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)


        x0 = self.g_s[:7](y_hat)  # Up to the 384-channel output
        
        # Split channels
        x1, x2 = torch.split(x0, [192, x0.size(1)-192], dim=1)  # Split into 192 channels each
        
        # First path: through IGDN and last conv
        x_conv = self.g_s[7:](x1)  # Process through IGDN and last conv
          # Second path: through spectral processing
        spectral_patches, patch_dim = split_into_patches(x2, num_patches=16)
        spectral_pooled = global_avg_pool_patches(spectral_patches)
        #spectral_pooled = spectral_pooled.view(batch_size, C+(C*(C-1))//2, 32, 16)
        spectral_pooled = spectral_pooled.view(batch_size, C+C, 32, 16)

        
        edge_features = spectral_pooled[:, (C):, :, :]
        encoded_node_features = self.patch_encoder(spectral_pooled[:, :(C), :, :])
        encoded_edge_features = self.patch_encoder(edge_features)
        
        recon_nodes, recon_edges, recon_adj, recon_T, recon_edge_adj = process_data(
            encoded_node_features, encoded_edge_features)
        
        recon_node, recon_edge = self.spectral_graph1(
            recon_nodes, recon_edges, recon_edge_adj, recon_adj, recon_T)
        
        # Only use node features for reconstruction
        rev_recon_nodes = self.reverse_patch_encoder(recon_node)
        rev_recon_patches = expand_to_patches(rev_recon_nodes, patch_dim)
        spectral_output = rearrange_patches_to_full(rev_recon_patches)
        
        # Process through grouped deconv to get 3-channel output
        x_spectral = self.grouped_deconv(spectral_output)
        
        # Average the two 3-channel outputs
        x_hat = (x_conv + x_spectral) / 2
        #x_hat = self.g_s(y_hat)
        

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    '''def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)'''

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.0.weight"].size(0)
        # M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(192, 320)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        ########################## spectral branch
        grouped_output = self.grouped_conv(x)
        batch_size, C, H, W = x.shape
        num_patches = 16  # You can change this to 4, 9, 25, 36, etc.

        # Split the features into patches
        patches, patch_dim = split_into_patches(grouped_output, num_patches)
        pooled_patches = global_avg_pool_patches(patches)

        grouped_pooled_patches = pooled_patches.view(batch_size, C, 32, 16)

        edge_features = compute_undirected_edge_features(grouped_pooled_patches)

     
        encoded_node_features = self.patch_encoder(grouped_pooled_patches)
        encoded_edge_features = self.patch_encoder(edge_features)


        nodes, edges, adj, T, edge_adj = process_data(encoded_node_features, encoded_edge_features)


        # initialize spectral_graph1 with parameters from grouped_output
     

        o_node, o_edge = self.spectral_graph1(nodes, edges, edge_adj, adj, T)
        
        

        rev_encoded_nodes = self.reverse_patch_encoder(o_node)
        rev_encoded_edges = self.reverse_patch_encoder(o_edge)

        output = torch.cat([rev_encoded_nodes, rev_encoded_edges], dim=1)

        rev_encoded_patches = expand_to_patches(output, patch_dim)
        rev_encoded_patches = rearrange_patches_to_full(rev_encoded_patches)


       # Process g_a with spectral injection
        x = self.g_a[:2](x)  # First conv and GDN
        x = torch.cat([x, rev_encoded_patches], dim=1)
        y = self.g_a[2:](x)  # Rest of g_a layers
    

        ########################## regular branch ##############################
        #y = self.g_a(x)
       # y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            if not support_slices:
                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            else:
                support_tensor = torch.cat(support_slices, dim=1)
                scale_sft = self.scale_conv[slice_index](support_tensor)
                shift_sft = self.shift_conv[slice_index](support_tensor)
                latent_mean_conditioned = latent_means * (scale_sft + 1) + shift_sft
                latent_scale_conditioned = latent_scales * (scale_sft + 1) + shift_sft
                mean_support =  latent_mean_conditioned #torch.cat([latent_mean_conditioned])+ support_slices, dim=1)
                scale_support = latent_scale_conditioned #torch.cat([latent_scale_conditioned]) + support_slices, dim=1)

           

            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            if not support_slices:
                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            else:
                support_tensor = torch.cat(support_slices, dim=1)
                scale_sft = self.scale_conv[slice_index](support_tensor)
                shift_sft = self.shift_conv[slice_index](support_tensor)
                latent_mean_conditioned = latent_means * (scale_sft + 1) + shift_sft
                latent_scale_conditioned = latent_scales * (scale_sft + 1) + shift_sft
                mean_support =  latent_mean_conditioned #torch.cat([latent_mean_conditioned])+ support_slices, dim=1)
                scale_support = latent_scale_conditioned #torch.cat([latent_scale_conditioned]) + support_slices, dim=1)




            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        x0 = self.g_s[:7](y_hat)  # Up to the 384-channel output
        batch_size = x0.shape[0]
        
        # Split channels
        x1, x2 = torch.split(x0, [192, x0.size(1)-192], dim=1)  # Split into 192 channels each
        
        # First path: through IGDN and last conv
        x_conv = self.g_s[7:](x1)  # Process through IGDN and last conv
        C=6
          # Second path: through spectral processing
        spectral_patches, patch_dim = split_into_patches(x2, num_patches=16)
        spectral_pooled = global_avg_pool_patches(spectral_patches)
        #spectral_pooled = spectral_pooled.view(batch_size, C+(C*(C-1))//2, 32, 16)
        spectral_pooled = spectral_pooled.view(batch_size, 2*C, 32, 16)

        
        edge_features = spectral_pooled[:, (C):, :, :]
        encoded_node_features = self.patch_encoder(spectral_pooled[:, :(C), :, :])
        encoded_edge_features = self.patch_encoder(edge_features)
        
        recon_nodes, recon_edges, recon_adj, recon_T, recon_edge_adj = process_data(
            encoded_node_features, encoded_edge_features)
        
        recon_node, recon_edge = self.spectral_graph1(
            recon_nodes, recon_edges, recon_edge_adj, recon_adj, recon_T)
        
        # Only use node features for reconstruction
        rev_recon_nodes = self.reverse_patch_encoder(recon_node)
        rev_recon_patches = expand_to_patches(rev_recon_nodes, patch_dim)
        spectral_output = rearrange_patches_to_full(rev_recon_patches)
        
        # Process through grouped deconv to get 3-channel output
        x_spectral = self.grouped_deconv(spectral_output)
        
        # Average the two 3-channel outputs
        x_hat = ((x_conv + x_spectral) / 2).clamp_(0, 1)
        #x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

in_channels = 6
out_channels_per_group = 32


class WinGraph_WA(WACNN):
    def __init__(self, N=192, M=320, knn = 9, graph_conv = 'transf_custom', heads = 8, use_edge_attr = False, dissimilarity = False, **kwargs):
        super().__init__(**kwargs)
        print(f'\n-----   WinGraph_WA model \n\
              knn: {knn} \n\
              graph-conv: {graph_conv}\n\
              heads: {heads} \n\
              use edge attribute: {use_edge_attr} \n\
              dissimilarity: {dissimilarity}\n\
              ------\n')


        


        self.grouped_conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=in_channels * out_channels_per_group,
                         kernel_size=5,
                         stride=2,
                         padding=2,
                         groups=in_channels
                         
            )

        self.grouped_deconv = nn.ConvTranspose2d(
                in_channels=in_channels * out_channels_per_group,
                out_channels=in_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                groups=in_channels
            )


        self.patch_encoder = PatchEncoder(in_features=32, out_features=128)

        nfeat_v = 128 #nodes.shape[1]  # Example: Number of vertex features
        nfeat_e = 128 #edges.shape[2]  # Example: Number of edge features
        nhid = 16
        nclass = 128  
        dropout = 0.5



        self.spectral_graph1 = Spectral_Graph(nfeat_v, nfeat_e, nhid, nclass, dropout)
        
        self.reverse_patch_encoder = ReversePatchEncoder(in_features=128, out_features=32)

        self.g_a = nn.Sequential(
            conv(6, N, kernel_size=5, stride=2),
            GDN(N),
            #conv(N+((in_channels*32))+((in_channels*(in_channels-1))*32)//2,N, kernel_size=5, stride=2),
            conv(N+((in_channels*32))+((in_channels*32)),N, kernel_size=5, stride=2),

            GDN(N),
            Win_GraphPyg(dim=N, window_size=8, knn=knn, conv=graph_conv, heads=heads, use_edge_attr=use_edge_attr, dissimilarity=dissimilarity),
            #Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_GraphPyg(dim=M, window_size=4, knn=knn, conv=graph_conv, heads=heads, use_edge_attr=use_edge_attr, dissimilarity=dissimilarity),
            #Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.g_s = nn.Sequential(
            #Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            Win_GraphPyg(dim=M, window_size=4, knn=knn, conv=graph_conv, heads=heads, use_edge_attr=use_edge_attr, dissimilarity=dissimilarity),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_GraphPyg(dim=N, window_size=8, knn=knn, conv=graph_conv, heads=heads, use_edge_attr=use_edge_attr, dissimilarity=dissimilarity),
            #Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            #deconv(N, N+((in_channels*32))+((in_channels*(in_channels-1))*32)//2,
                #kernel_size=5, stride=2),
            deconv(N, N+((in_channels*32))+((in_channels*32)),
                kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 6, kernel_size=5, stride=2),
        )



if __name__ == '__main__':
     
    model = WinGraph_WA(
        knn = 9,
        graph_conv = 'transf_custom', # '',
        heads = 8, 
        use_edge_attr = True,
        dissimilarity = False 
    ).to('cuda')
    model.update(force = True)
    model.eval()

    x = torch.rand((1, 3, 512, 768)).to('cuda')

    print(model(x)['x_hat'].shape)