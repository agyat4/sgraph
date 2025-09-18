import math
import torch
from torch import nn
import torch.nn.functional as F


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


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def knn_sparse(x, k=16, relative_pos=None, dissimilarity = False, loop = False):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                if(dissimilarity):
                    _, nn_idx_part = torch.topk(dist, k=k)
                else:
                    if(loop):
                        _, nn_idx_part = torch.topk(-dist, k=k)
                    else:
                        _, nn_idx_part = torch.topk(-dist, k=k+1)
                        nn_idx = nn_idx[:,:,1:]


                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            
            if(dissimilarity):
                _, nn_idx = torch.topk(dist, k=k) # b, n, k
            else:
                if(loop):
                    vals, nn_idx = torch.topk(-dist, k=k) # b, n, k
                else:
                    vals, nn_idx = torch.topk(-dist, k=k+1) # b, n, k
                    nn_idx = nn_idx[:,:,1:]
                    vals = vals[:,:,1:]
                #print(vals)

        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1) # b, n, k
        
        # batch count
        a = [(torch.ones((1,n_points,k), dtype=torch.int)*(i*n_points)).to(device=x.device) for i in range(batch_size)]
        batch_counter = torch.Tensor((batch_size, n_points, k)).resize_(0).to(device=x.device)
        torch.cat(a, out=batch_counter)

        center_idx = center_idx+batch_counter
        nn_idx = nn_idx + batch_counter


        # to sparse
        center_idx = center_idx.reshape(-1) # b*n*k
        nn_idx = nn_idx.reshape(-1) # b*n*k



    return torch.stack((nn_idx, center_idx), dim=0)


def xy_knn_sparse(x, y, k=16, relative_pos=None, dissimilarity = False, loop = False):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        
        batch_size, n_points, n_dims = x.shape
        _,n_points_y,_ = y.shape

        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos

        if(dissimilarity):
            _, nn_idx = torch.topk(dist, k=k)
        else:
            if(loop):
                _, nn_idx = torch.topk(-dist, k=k)
            else:
                _, nn_idx = torch.topk(-dist, k=k+1)
                nn_idx = nn_idx[:,:,1:]


        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)

        # batch count
        a = [(torch.ones((1,n_points,k), dtype=torch.int)*(i*n_points)).to(device=x.device) for i in range(batch_size)]
        batch_counter = torch.Tensor((batch_size, n_points, k)).resize_(0).to(device=x.device)
        torch.cat(a, out=batch_counter)

        a = [(torch.ones((1,n_points_y,k), dtype=torch.int)*(i*n_points_y)).to(device=x.device) for i in range(batch_size)]
        batch_counter_y = torch.Tensor((batch_size, n_points_y, k)).resize_(0).to(device=x.device)
        torch.cat(a, out=batch_counter_y)

        center_idx = center_idx+batch_counter
        nn_idx = nn_idx + batch_counter_y


        # to sparse
        center_idx = center_idx.reshape(-1) # b*n*k
        nn_idx = nn_idx.reshape(-1) # b*n*k


    return torch.stack((nn_idx, center_idx), dim=0)





class SparseKnnGraph(nn.Module):
    def __init__(self, k=9, dissimilarity = False, loop = False):
        super(SparseKnnGraph, self).__init__()
        self.k = k
        self.dissimilarity = dissimilarity
        self.loop = loop
        print(f'knn: {self.k} - dissimilarity: {self.dissimilarity}')


    def forward(self, x,limit_adj = None):
        B,C,_,_ = x.shape

        x = x.reshape(B, C, -1, 1).contiguous()
            
        #### normalize
        x = F.normalize(x, p=2.0, dim=1)
        ####
        edge_index = knn_sparse(x, self.k, limit_adj, dissimilarity=self.dissimilarity, loop=self.loop)
        return edge_index.long()
