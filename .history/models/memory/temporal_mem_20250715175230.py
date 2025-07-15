"""
Temporal Memory Module for DMIAD
Adapted from video anomaly detection for image anomaly detection
Processes feature channels as temporal-like patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


class MemoryUnit(nn.Module):
    """
    Memory Unit for temporal-like processing
    Adapted from video anomaly detection memory module
    """
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.005):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres = shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        Args:
            input: [N*H*W, C] - flattened spatial features
        Returns:
            dict with output, attention weights, and memory similarities
        """
        # Normalize input features and memory weights
        input = F.normalize(input, dim=1)
        weight_normal = F.normalize(self.weight, dim=1)
        
        # Compute attention weights: query features vs memory items
        att_weight = F.linear(input, weight_normal)  # [N*H*W, M]
        
        # Memory-to-memory alignment for contrastive learning
        mem_fea_align = F.linear(weight_normal, weight_normal)  # [M, M]
        
        # Apply softmax to get attention distribution
        att_weight = F.softmax(att_weight, dim=1)  # [N*H*W, M]
        
        # Apply shrinkage if needed for sparsity
        if self.shrink_thres > 0:
            att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        
        # Generate output: attention-weighted memory items
        mem_trans = self.weight.permute(1, 0)  # [C, M]
        output = F.linear(att_weight, mem_trans)  # [N*H*W, C]
        
        return {
            'output': output, 
            'att': att_weight, 
            'mem_fea_align': mem_fea_align
        }

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim
        )


class MemModule(nn.Module):
    """
    Memory Module for temporal-like processing
    Processes spatial features in a temporal-like manner
    """
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, shrink_thres)

    def forward(self, input):
        """
        Process features through memory module
        
        Args:
            input: [N, C, H, W] - input features
        Returns:
            dict with memory processing results
        """
        N, C, H, W = input.shape
        
        # Flatten spatial dimensions: [N, C, H, W] -> [N*H*W, C]
        x = input.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        x = x.view(N * H * W, C)  # [N*H*W, C]
        
        # Process through memory unit
        y_and = self.memory(x)
        
        # Extract outputs
        y = y_and['output']      # [N*H*W, C]
        att = y_and['att']       # [N*H*W, M]
        mem_fea_align = y_and['mem_fea_align']  # [M, M]
        
        # Reshape back to spatial format
        y = y.view(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        att = att.view(N, H, W, self.mem_dim).permute(0, 3, 1, 2)  # [N, M, H, W]
        
        return {
            'output': y,
            'att': att,
            'mem_fea_align': mem_fea_align,
            'input_flat': x,  # Keep flattened input for loss computation
            'raw_attention': y_and['att']  # Raw attention [N*H*W, M]
        }


class TemporalMemoryWithMLP(nn.Module):
    """
    Enhanced Temporal Memory Module with shared MLP processing
    Similar to spatial memory but for temporal-like patterns
    """
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, 
                 use_shared_mlp=False, normalize_memory=True, normalize_query=True):
        super(TemporalMemoryWithMLP, self).__init__()
        
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.use_shared_mlp = use_shared_mlp
        self.normalize_memory = normalize_memory
        self.normalize_query = normalize_query
        
        # Memory parameters
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        
        # Shared MLP for processing memory items (like in spatial memory)
        if self.use_shared_mlp:
            self.shared_mlp = nn.Sequential(
                nn.Linear(fea_dim, fea_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(fea_dim // 2, fea_dim),
                nn.ReLU(inplace=True)
            )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        """
        Enhanced memory processing with configurable options
        
        Args:
            input: [N, C, H, W] - input features
        Returns:
            dict with enhanced memory outputs
        """
        N, C, H, W = input.shape
        
        # Flatten spatial dimensions
        x = input.permute(0, 2, 3, 1).contiguous().view(N * H * W, C)
        
        # Configurable normalization for queries
        if self.normalize_query:
            x_norm = F.normalize(x, dim=1)
        else:
            x_norm = x
        
        # Configurable normalization for memory
        if self.normalize_memory:
            weight_norm = F.normalize(self.weight, dim=1)
        else:
            weight_norm = self.weight
        
        # Apply shared MLP to memory weights if enabled
        if self.use_shared_mlp:
            processed_memory = self.shared_mlp(weight_norm)
            if self.normalize_memory:
                processed_memory = F.normalize(processed_memory, dim=1)
        else:
            processed_memory = weight_norm
        
        # Compute attention weights
        att_weight = F.linear(x_norm, processed_memory)  # [N*H*W, M]
        
        # Memory-to-memory similarities
        mem_fea_align = F.linear(processed_memory, processed_memory)  # [M, M]
        
        # Softmax attention
        att_weight = F.softmax(att_weight, dim=1)
        
        # Apply shrinkage if needed
        if self.shrink_thres > 0:
            att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        
        # Generate output
        mem_trans = processed_memory.permute(1, 0)  # [C, M]
        output = F.linear(att_weight, mem_trans)  # [N*H*W, C]
        
        # Reshape back to spatial format
        output = output.view(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        att_spatial = att_weight.view(N, H, W, self.mem_dim).permute(0, 3, 1, 2)  # [N, M, H, W]
        
        return {
            'output': output,
            'att': att_spatial,
            'mem_fea_align': mem_fea_align,
            'processed_memory': processed_memory,
            'input_flat': x,
            'raw_attention': att_weight
        }


def build_temporal_memory(config):
    """Build temporal memory module based on configuration"""
    
    mem_config = config.MODEL.MEMORY
    
    if mem_config.get('USE_ENHANCED_TEMPORAL', False):
        # Use enhanced temporal memory with MLP
        memory_module = TemporalMemoryWithMLP(
            mem_dim=mem_config.TEMPORAL_DIM,
            fea_dim=512,  # This should match the projected feature dimension
            shrink_thres=mem_config.SHRINK_THRES,
            use_shared_mlp=mem_config.get('USE_SHARED_MLP', False),
            normalize_memory=mem_config.get('NORMALIZE_MEMORY', True),
            normalize_query=mem_config.get('NORMALIZE_QUERY', True)
        )
    else:
        # Use standard temporal memory
        memory_module = MemModule(
            mem_dim=mem_config.TEMPORAL_DIM,
            fea_dim=512,  # This should match the projected feature dimension
            shrink_thres=mem_config.SHRINK_THRES
        )
    
    return memory_module


# Utility functions for memory analysis
def compute_memory_utilization(attention_weights, threshold=0.01):
    """
    Compute memory utilization statistics
    
    Args:
        attention_weights: [N*H*W, M] or [N, M, H, W] - attention weights
        threshold: minimum attention to consider a memory item as "used"
    
    Returns:
        dict with utilization statistics
    """
    if attention_weights.dim() == 4:
        # Convert [N, M, H, W] to [N*H*W, M]
        N, M, H, W = attention_weights.shape
        attention_weights = attention_weights.permute(0, 2, 3, 1).contiguous().view(-1, M)
    
    # Compute utilization statistics
    max_attention = torch.max(attention_weights, dim=0)[0]  # [M]
    mean_attention = torch.mean(attention_weights, dim=0)  # [M]
    
    # Count how many memory items are actively used
    active_memories = (max_attention > threshold).sum().item()
    total_memories = attention_weights.shape[1]
    
    utilization_stats = {
        'active_memories': active_memories,
        'total_memories': total_memories,
        'utilization_rate': active_memories / total_memories,
        'max_attention_mean': float(torch.mean(max_attention)),
        'max_attention_std': float(torch.std(max_attention)),
        'mean_attention_mean': float(torch.mean(mean_attention)),
        'attention_entropy': float(compute_attention_entropy(attention_weights))
    }
    
    return utilization_stats


def compute_attention_entropy(attention_weights):
    """
    Compute entropy of attention distribution
    Higher entropy indicates more diverse memory usage
    """
    # Average attention over spatial locations
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.permute(0, 2, 3, 1).contiguous().view(-1, attention_weights.shape[1])
    
    avg_attention = torch.mean(attention_weights, dim=0)  # [M]
    
    # Compute entropy
    epsilon = 1e-12
    entropy = -torch.sum(avg_attention * torch.log(avg_attention + epsilon))
    
    return entropy


# Example usage and testing
if __name__ == "__main__":
    # Test temporal memory module
    batch_size, channels, height, width = 2, 512, 16, 16
    mem_dim = 1000
    
    # Create test input
    test_input = torch.randn(batch_size, channels, height, width)
    
    # Test standard temporal memory
    temporal_mem = MemModule(mem_dim=mem_dim, fea_dim=channels)
    
    output = temporal_mem(test_input)
    
    print("Temporal Memory Module Test:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output['output'].shape}")
    print(f"Attention shape: {output['att'].shape}")
    print(f"Memory alignment shape: {output['mem_fea_align'].shape}")
    
    # Test enhanced temporal memory
    enhanced_mem = TemporalMemoryWithMLP(
        mem_dim=mem_dim, 
        fea_dim=channels,
        use_shared_mlp=True,
        normalize_memory=True,
        normalize_query=True
    )
    
    enhanced_output = enhanced_mem(test_input)
    
    print(f"\nEnhanced Temporal Memory Test:")
    print(f"Output shape: {enhanced_output['output'].shape}")
    print(f"Processed memory shape: {enhanced_output['processed_memory'].shape}")
    
    # Test memory utilization analysis
    utilization = compute_memory_utilization(output['att'])
    print(f"\nMemory Utilization Analysis:")
    print(f"Active memories: {utilization['active_memories']}/{utilization['total_memories']}")
    print(f"Utilization rate: {utilization['utilization_rate']:.3f}")
    print(f"Attention entropy: {utilization['attention_entropy']:.3f}")