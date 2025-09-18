import torch
from timm.models.layers import to_2tuple
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size=7):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """

    x = x.transpose(1,2).transpose(2,3)
    B, H, W, C = x.shape
    windows = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    
    windows = windows.transpose(2,3).transpose(1,2)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    windows = windows.transpose(1,2).transpose(2,3)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    x = x.transpose(2,3).transpose(1,2)

    return x

