"""
Spatial Memory Module for DMIAD
Processes spatial patterns across channels
Complementary to temporal memory processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


class PositionalEncoding2D(nn.Module):
    """2D Positional Encoding for spatial features"""
    def __init__(self, channels, height, width, temperature=10000):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        self.temperature = temperature
        self.register_buffer('pos_embed', self._generate_pos_embed(channels, height, width))
    
    def _generate_pos_embed(self, channels, height, width):
        """Generate 2D positional embeddings"""
        # Create coordinate grids
        y_pos = torch.arange(height, dtype=torch.float32).unsqueeze(1).repeat(1, width)
        x_pos = torch.arange(width, dtype=torch.float32).unsqueeze(0).repeat(height, 1)
        
        # Normalize coordinates to [0, 1]
        if height > 1:
            y_pos = y_pos / (height - 1)
        if width > 1:
            x_pos = x_pos / (width - 1)
        
        # Calculate dimension for each coordinate
        dim_t = torch.arange(channels // 4, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (channels // 4))
        
        # Generate positional embeddings
        pos_x = x_pos.unsqueeze(-1) / dim_t
        pos_y = y_pos.unsqueeze(-1) / dim_t
        
        # Apply sin/cos
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        # Concatenate x and y positional embeddings
        pos_embed = torch.cat((pos_y, pos_x), dim=2)  # [H, W, channels//2]
        
        # If channels is odd, pad with zeros
        if channels % 2 == 1:
            pos_embed = torch.cat([pos_embed, torch.zeros(height, width, 1)], dim=2)
        
        return pos_embed.permute(2, 0, 1)  # [C, H, W]
    
    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        Returns:
            x with positional encoding added: [N, C, H, W]
        """
        return x + self.pos_embed.unsqueeze(0)


class SpatialMemoryUnit(nn.Module):
    """
    Memory unit for spatial pattern processing
    Each memory item stores a spatial pattern template
    """
    def __init__(self, mem_dim, spatial_dim, shrink_thres=0.005, 
                 normalize_memory=True, normalize_query=True, use_shared_mlp=False):
        super(SpatialMemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.spatial_dim = spatial_dim  # H*W
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.spatial_dim))  # M x (H*W)
        self.shrink_thres = shrink_thres
        self.normalize_memory = normalize_memory
        self.normalize_query = normalize_query
        self.use_shared_mlp = use_shared_mlp
        
        # Shared MLP for processing memory spatial patterns
        if self.use_shared_mlp:
            self.shared_mlp = nn.Sequential(
                nn.Linear(spatial_dim, spatial_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(spatial_dim // 2, spatial_dim),
                nn.ReLU(inplace=True)
            )