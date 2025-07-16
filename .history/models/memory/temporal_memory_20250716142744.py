"""
Temporal Memory Module
Pixel-wise processing - mỗi memory slot học pattern của 1 pixel
Memory shape: [mem_size, channels]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # self.register_buffer('memory', torch.randn(mem_dim, fea_dim))
        self.memory = nn.Parameter(torch.randn(mem_dim, fea_dim))
        nn.init.normal_(self.memory, mean=0, std=0.1)
        self.memory.data = F.normalize(self.memory.data, dim=1) 
        # self.memory = F.normalize(self.memory, dim=1)
        
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


def build_temporal_memory(config):
    """Build temporal memory module"""
    return TemporalMemoryModule(
        mem_dim=config.TEMPORAL_DIM,
        fea_dim=config.FEATURE_DIM,
        shrink_thres=config.SHRINK_THRES
    )


# Test function
if __name__ == "__main__":
    # Test temporal memory
    temp_mem = TemporalMemoryModule(mem_dim=2000, fea_dim=512)
    
    # Test input
    x = torch.randn(2, 512, 32, 32)
    
    # Test forward pass
    with torch.no_grad():
        temp_out = temp_mem(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Temporal output shape: {temp_out['output'].shape}")
        print(f"Temporal memory shape: {temp_mem.memory.shape}")  # [2000, 512]
        print(f"Temporal attention shape: {temp_out['att'].shape}")
        
    print("Temporal memory test completed!")