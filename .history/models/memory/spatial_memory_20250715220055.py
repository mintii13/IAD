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
    Mỗi memory slot học pattern của 1 channel (giữ nguyên spatial dimensions)
    Memory shape: [mem_size, H, W] - giữ nguyên spatial structure
    """
    def __init__(self, mem_dim, height, width, shrink_thres=0.0025):
        super(SpatialMemoryModule, self).__init__()
        
        self.mem_dim = mem_dim  # Number of memory slots
        self.height = height    # Spatial height
        self.width = width      # Spatial width
        self.shrink_thres = shrink_thres
        
        # Memory bank H: [mem_size, H, W] - giữ nguyên spatial structure
        self.register_buffer('memory', torch.randn(mem_dim, height, width))
        self.memory = F.normalize(self.memory.view(mem_dim, -1), dim=1).view(mem_dim, height, width)
        
        # Shared MLP cho spatial processing - hoạt động trên spatial patterns
        hidden_dim = (height * width) // 2
        self.shared_mlp = nn.Sequential(
            nn.Linear(height * width, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, height * width),
            nn.ReLU(inplace=True)
        )

    def compute_ssim_similarity(self, query_channels, memory_patterns):
        """
        Compute SSIM similarity between query channels and memory patterns
        
        Args:
            query_channels: [N*C, H, W] - flattened query channels
            memory_patterns: [mem_dim, H, W] - memory spatial patterns
        Returns:
            similarity: [N*C, mem_dim] - SSIM similarities
        """
        N_C, H, W = query_channels.shape
        mem_dim = memory_patterns.shape[0]
        
        # Flatten spatial dimensions for easier computation
        query_flat = query_channels.view(N_C, H * W)  # [N*C, H*W]
        memory_flat = memory_patterns.view(mem_dim, H * W)  # [mem_dim, H*W]
        
        # Compute means
        query_mean = torch.mean(query_flat, dim=1, keepdim=True)  # [N*C, 1]
        memory_mean = torch.mean(memory_flat, dim=1, keepdim=True)  # [mem_dim, 1]
        
        # Compute variances
        query_var = torch.var(query_flat, dim=1, keepdim=True)  # [N*C, 1]
        memory_var = torch.var(memory_flat, dim=1, keepdim=True)  # [mem_dim, 1]
        
        # Center the data
        query_centered = query_flat - query_mean
        memory_centered = memory_flat - memory_mean
        
        # Compute covariance (cross-correlation)
        covariance = torch.mm(query_centered, memory_centered.t()) / (H * W - 1)  # [N*C, mem_dim]
        
        # SSIM formula components
        c1, c2 = 0.01, 0.03
        
        # Numerator: (2*mu1*mu2 + c1) * (2*cov + c2)
        mean_product = torch.mm(query_mean, memory_mean.t())  # [N*C, mem_dim]
        numerator = (2 * mean_product + c1) * (2 * covariance + c2)
        
        # Denominator: (mu1^2 + mu2^2 + c1) * (var1 + var2 + c2)
        mean_sum = query_mean**2 + memory_mean.t()**2  # [N*C, mem_dim]
        var_sum = query_var + memory_var.t()  # [N*C, mem_dim]
        denominator = (mean_sum + c1) * (var_sum + c2)
        
        # SSIM similarity
        ssim = numerator / (denominator + 1e-8)
        
        return ssim

    def forward(self, input):
        """
        Args:
            input: [N, C, H, W] feature maps
        Returns:
            dict với output và memory information
        """
        s = input.data.shape
        N, C, H, W = s
        
        # Channel-wise processing: [N, C, H, W] → [N*C, H, W]
        # Mỗi slice là spatial pattern của 1 channel
        input_channels = input.view(N * C, H, W)  # [N*C, H, W]
        
        # Process query channels through MLP (flatten → MLP → reshape)
        input_flat = input_channels.view(N * C, H * W)  # [N*C, H*W]
        input_processed_flat = self.shared_mlp(input_flat)  # [N*C, H*W]
        input_processed = input_processed_flat.view(N * C, H, W)  # [N*C, H, W]
        
        # Process memory through MLP
        memory_flat = self.memory.view(self.mem_dim, H * W)  # [mem_dim, H*W]
        memory_processed_flat = self.shared_mlp(memory_flat)  # [mem_dim, H*W]
        memory_processed = memory_processed_flat.view(self.mem_dim, H, W)  # [mem_dim, H, W]
        
        # Normalize processed features
        input_norm = F.normalize(input_processed.view(N * C, H * W), dim=1).view(N * C, H, W)
        memory_norm = F.normalize(memory_processed.view(self.mem_dim, H * W), dim=1).view(self.mem_dim, H, W)
        
        # Compute SSIM similarity
        ssim_similarity = self.compute_ssim_similarity(input_norm, memory_norm)  # [N*C, mem_dim]
        
        # Apply softmax to get attention weights
        att_weight = F.softmax(ssim_similarity, dim=1)  # [N*C, mem_dim]
        
        # Apply shrinkage
        if self.shrink_thres > 0:
            att_weight = F.relu(att_weight - self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        
        # Retrieve from memory: attention × memory
        # att_weight: [N*C, mem_dim], memory_norm: [mem_dim, H, W]
        memory_flat_norm = memory_norm.view(self.mem_dim, H * W)  # [mem_dim, H*W]
        output_flat = torch.mm(att_weight, memory_flat_norm)  # [N*C, H*W]
        output_channels = output_flat.view(N * C, H, W)  # [N*C, H, W]
        
        # Reshape back to original format
        output = output_channels.view(N, C, H, W)  # [N, C, H, W]
        
        # Reshape attention for compatibility
        att_reshaped = att_weight.view(N, C, self.mem_dim)
        att_channel_avg = att_reshaped.mean(dim=1)  # [N, mem_dim]
        att_spatial = att_channel_avg.unsqueeze(-1).unsqueeze(-1).expand(N, self.mem_dim, H, W)
        
        return {
            'output': output,
            'att': att_spatial,
            'input_flat': input_flat,
            'memory': memory_norm,
            'raw_attention': att_weight,
            'ssim_similarity': ssim_similarity
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
    # Calculate spatial dimensions from input size
    # Assuming input after backbone is [B, C, H/8, W/8] due to stride 8
    # For 288x288 input → 36x36 feature maps
    spatial_size = config.DATASET.CROP_SIZE // 8  # 288 // 8 = 36
    
    return SpatialMemoryModule(
        mem_dim=config.SPATIAL_DIM,
        height=spatial_size,
        width=spatial_size,
        shrink_thres=config.SHRINK_THRES
    )


# Test function
if __name__ == "__main__":
    # Test temporal memory
    temp_mem = TemporalMemoryModule(mem_dim=2000, fea_dim=512)
    
    # Test spatial memory  
    spa_mem = SpatialMemoryModule(mem_dim=2000, height=32, width=32)
    
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
        print(f"Spatial memory shape: {spa_mem.memory.shape}")    # [2000, 32, 32]
        
    print("Memory modules test completed!")