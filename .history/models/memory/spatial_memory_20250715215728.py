"""
Temporal và Spatial Memory Modules - đúng theo pipeline
- Temporal: pixel-wise, memory shape [mem_size, channel]  
- Spatial: channel-wise, memory shape [mem_size, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


class TemporalMemoryModule(nn.Module):
    """
    Temporal Memory - Pixel-wise processing
    Mỗi memory slot học pattern của 1 pixel (across channels)
    Memory shape: [mem_size, channels]
    """
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(TemporalMemoryModule, self).__init__()
        
        self.mem_dim = mem_dim  # Number of memory slots
        self.fea_dim = fea_dim  # Feature dimension (channels)
        self.shrink_thres = shrink_thres
        
        # Memory bank C: [mem_size, channels]
        self.register_buffer('memory', torch.randn(mem_dim, fea_dim))
        self.memory = F.normalize(self.memory, dim=1)
        
        # Shared MLP cho temporal processing
        self.shared_mlp = nn.Sequential(
            nn.Linear(fea_dim, fea_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fea_dim // 2, fea_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        """
        Args:
            input: [N, C, H, W] feature maps
        Returns:
            dict với output và memory information
        """
        s = input.data.shape
        N, C, H, W = s
        
        # Pixel-wise flatten: [N, C, H, W] → [N*H*W, C]
        # Mỗi row là feature vector của 1 pixel
        input_flat = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [N*H*W, C]
        
        # Normalize input
        input_norm = F.normalize(input_flat, dim=1)
            
        # Process memory through shared MLP
        memory_processed = self.shared_mlp(self.memory)  # [mem_dim, C]
        memory_norm = F.normalize(memory_processed, dim=1)
        
        # Compute attention weights: softmax(query • memory)
        att_weight = torch.mm(input_norm, memory_norm.t())  # [N*H*W, mem_dim]
        att_weight = F.softmax(att_weight, dim=1)
        
        # Apply shrinkage
        if self.shrink_thres > 0:
            att_weight = F.relu(att_weight - self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        
        # Retrieve from memory: attention × memory
        output_flat = torch.mm(att_weight, memory_norm)  # [N*H*W, C]
        
        # Reshape back to spatial format
        output = output_flat.view(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        
        # Reshape attention for compatibility  
        att_reshaped = att_weight.view(N, H, W, self.mem_dim).permute(0, 3, 1, 2)  # [N, mem_dim, H, W]
        
        return {
            'output': output,
            'att': att_reshaped,
            'input_flat': input_flat,
            'memory': memory_norm,
            'raw_attention': att_weight
        }


class SpatialMemoryModule(nn.Module):
    """
    Spatial Memory - Channel-wise processing  
    Mỗi memory slot học pattern của 1 channel (across spatial locations)
    Memory shape: [mem_size, H, W]
    """
    def __init__(self, mem_dim, spatial_dim, shrink_thres=0.0025):
        super(SpatialMemoryModule, self).__init__()
        
        self.mem_dim = mem_dim  # Number of memory slots
        self.spatial_dim = spatial_dim  # H * W
        self.shrink_thres = shrink_thres
        
        # Memory bank H: [mem_size, spatial_dim] where spatial_dim = H*W
        self.register_buffer('memory', torch.randn(mem_dim, spatial_dim))
        self.memory = F.normalize(self.memory, dim=1)
        
        # Shared MLP cho spatial processing
        self.shared_mlp = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_dim // 2, spatial_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        """
        Args:
            input: [N, C, H, W] feature maps
        Returns:
            dict với output và memory information
        """
        s = input.data.shape
        N, C, H, W = s
        
        # Channel-wise flatten: [N, C, H, W] → [N*C, H*W]
        # Mỗi row là spatial pattern của 1 channel
        input_flat = input.view(N * C, H * W)  # [N*C, H*W]
        
        # Normalize input
        input_norm = F.normalize(input_flat, dim=1)
            
        # Process memory through shared MLP
        memory_processed = self.shared_mlp(self.memory)  # [mem_dim, H*W]
        memory_norm = F.normalize(memory_processed, dim=1)
        
        # Compute attention weights: softmax(SSIM(query, memory))
        # Simplified SSIM as cosine similarity
        att_weight = torch.mm(input_norm, memory_norm.t())  # [N*C, mem_dim]
        att_weight = F.softmax(att_weight, dim=1)
        
        # Apply shrinkage
        if self.shrink_thres > 0:
            att_weight = F.relu(att_weight - self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        
        # Retrieve from memory: attention × memory
        output_flat = torch.mm(att_weight, memory_norm)  # [N*C, H*W]
        
        # Reshape back to channel format
        output = output_flat.view(N, C, H, W)  # [N, C, H, W]
        
        # Reshape attention for compatibility
        att_reshaped = att_weight.view(N, C, self.mem_dim)
        att_channel_avg = att_reshaped.mean(dim=1)  # [N, mem_dim]
        att_spatial = att_channel_avg.unsqueeze(-1).unsqueeze(-1).expand(N, self.mem_dim, H, W)
        
        return {
            'output': output,
            'att': att_spatial,
            'input_flat': input_flat,
            'memory': memory_norm,
            'raw_attention': att_weight
        }


def build_temporal_memory(config):
    """Build temporal memory module"""
    return TemporalMemoryModule(
        mem_dim=config.TEMPORAL_DIM,
        fea_dim=config.FEATURE_DIM,
        shrink_thres=config.SHRINK_THRES
    )


def build_spatial_memory(config):
    """Build spatial memory module"""
    # Calculate spatial dimension from input size
    # Assuming input after backbone is [B, C, H/8, W/8] due to stride 8
    # For 256x256 input → 32x32 feature maps → spatial_dim = 32*32 = 1024
    spatial_dim = (config.DATASET.CROP_SIZE // 8) ** 2  # H*W after backbone
    
    return SpatialMemoryModule(
        mem_dim=config.SPATIAL_DIM,
        spatial_dim=spatial_dim,
        shrink_thres=config.SHRINK_THRES
    )


# Test function
if __name__ == "__main__":
    # Test temporal memory
    temp_mem = TemporalMemoryModule(mem_dim=2000, fea_dim=512)
    
    # Test spatial memory  
    spa_mem = SpatialMemoryModule(mem_dim=2000, spatial_dim=32*32)  # 32x32 spatial
    
    # Test input
    x = torch.randn(2, 512, 32, 32)
    
    # Test forward pass
    with torch.no_grad():
        temp_out = temp_mem(x)
        spa_out = spa_mem(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Temporal output shape: {temp_out['output'].shape}")
        print(f"Spatial output shape: {spa_out['output'].shape}")
        print(f"Temporal memory shape: {temp_mem.memory.shape}")  # [2000, 512]
        print(f"Spatial memory shape: {spa_mem.memory.shape}")    # [2000, 1024]
        
    print("Memory modules test completed!")