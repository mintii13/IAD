"""
Temporal Memory Module - theo pipeline diagram
Xử lý features như temporal patterns (channel-wise processing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


class MLPModule(nn.Module):
    """MLP module for memory processing như trong pipeline"""
    def __init__(self, input_dim, hidden_dim, output_dim, normalize=True):
        super(MLPModule, self).__init__()
        self.normalize = normalize
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return self.mlp(x)


class TemporalMemoryModule(nn.Module):
    """
    Temporal Memory Module theo pipeline diagram
    
    Pipeline: Z -> MLP_temp -> Normalize -> softmax(Z_temp • M_temp) -> W_temp -> Memsize -> Z_temp
    """
    def __init__(self, feature_dim, memory_size, memory_dim, shrink_thres=0.0025):
        super(TemporalMemoryModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.shrink_thres = shrink_thres
        
        # Memory bank C (như trong diagram)
        self.register_buffer('C', torch.randn(memory_size, memory_dim))
        self.C = F.normalize(self.C, dim=1)
        
        # MLP modules với shared weights (như trong diagram)
        self.mlp_temp_query = MLPModule(feature_dim, feature_dim//2, memory_dim, normalize=True)
        self.mlp_temp_memory = MLPModule(memory_dim, memory_dim//2, memory_dim, normalize=True)
        
        # Memsize operation (transform memory về feature space)
        self.memsize_temp = nn.Linear(memory_dim, feature_dim)
        
    def forward(self, z):
        """
        Temporal memory processing theo pipeline diagram
        
        Args:
            z: Input features [B, C, H, W]
        Returns:
            dict với output và attention information
        """
        B, C, H, W = z.shape
        
        # Flatten spatial dimensions cho temporal processing
        # [B, C, H, W] -> [B*H*W, C] - mỗi row là 1 pixel's features
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        # === PIPELINE STEP 1: Process query qua MLP ===
        Z_temp = self.mlp_temp_query(z_flat)  # [B*H*W, memory_dim]
        
        # === PIPELINE STEP 2: Process memory qua MLP ===  
        M_temp = self.mlp_temp_memory(self.C)  # [memory_size, memory_dim]
        
        # === PIPELINE STEP 3: Compute attention weights ===
        # softmax(Z_temp • M_temp) trong diagram
        attention_logits = torch.matmul(Z_temp, M_temp.t())  # [B*H*W, memory_size]
        W_temp = F.softmax(attention_logits, dim=1)  # [B*H*W, memory_size]
        
        # Shrinkage operation (optional)
        if self.shrink_thres > 0:
            W_temp = F.relu(W_temp - self.shrink_thres)
            W_temp = F.normalize(W_temp, p=1, dim=1)
        
        # === PIPELINE STEP 4: Memory retrieval ===
        # W_temp × M_temp
        retrieved_memory = torch.matmul(W_temp, M_temp)  # [B*H*W, memory_dim]
        
        # === PIPELINE STEP 5: Memsize operation ===
        # Transform memory về feature space
        Z_temp_out = self.memsize_temp(retrieved_memory)  # [B*H*W, feature_dim]
        
        # === PIPELINE STEP 6: Reshape về spatial format ===
        Z_temp_final = Z_temp_out.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return {
            'output': Z_temp_final,  # Reconstructed features
            'attention': W_temp.view(B, H, W, self.memory_size),  # Attention maps
            'retrieved_memory': retrieved_memory.view(B, H, W, self.memory_dim),
            'processed_query': Z_temp.view(B, H, W, self.memory_dim),
            'processed_memory': M_temp,  # Processed memory bank
            'raw_attention': W_temp  # Raw attention [B*H*W, memory_size]
        }


class TemporalMemoryWithSharedMLP(nn.Module):
    """
    Temporal Memory với shared MLP processing
    Tương thích với code gốc nhưng theo pipeline structure
    """
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, 
                 normalize_memory=True, normalize_query=True, use_shared_mlp=True):
        super(TemporalMemoryWithSharedMLP, self).__init__()
        
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.normalize_memory = normalize_memory
        self.normalize_query = normalize_query
        self.use_shared_mlp = use_shared_mlp
        
        # Memory bank
        self.register_buffer('memory', torch.randn(mem_dim, fea_dim))
        self.memory = F.normalize(self.memory, dim=1)
        
        # Shared MLP cho memory processing
        if self.use_shared_mlp:
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
        
        # Flatten spatial dimensions
        input_flat = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [N*H*W, C]
        
        # Normalize input nếu cần
        if self.normalize_query:
            input_norm = F.normalize(input_flat, dim=1)
        else:
            input_norm = input_flat
            
        # Process memory
        memory_to_use = self.memory
        if self.use_shared_mlp:
            memory_to_use = self.shared_mlp(memory_to_use)
            
        # Normalize memory nếu cần  
        if self.normalize_memory:
            memory_norm = F.normalize(memory_to_use, dim=1)
        else:
            memory_norm = memory_to_use
        
        # Compute attention weights
        att_weight = torch.mm(input_norm, memory_norm.t())  # [N*H*W, mem_dim]
        att_weight = F.softmax(att_weight, dim=1)
        
        # Apply shrinkage
        if self.shrink_thres > 0:
            att_weight = F.relu(att_weight - self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        
        # Retrieve memory
        output_flat = torch.mm(att_weight, memory_norm)  # [N*H*W, C]
        
        # Reshape về spatial format
        output = output_flat.view(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        
        # Reshape attention cho compatibility
        att_reshaped = att_weight.view(N, H, W, self.mem_dim).permute(0, 3, 1, 2)  # [N, mem_dim, H, W]
        
        return {
            'output': output,
            'att': att_reshaped,
            'input_flat': input_flat,
            'memory': memory_norm,
            'raw_attention': att_weight
        }


def build_temporal_memory(config):
    """
    Build temporal memory module based on config
    """
    if hasattr(config, 'USE_SHARED_MLP') and config.USE_SHARED_MLP:
        return TemporalMemoryWithSharedMLP(
            mem_dim=config.TEMPORAL_DIM,
            fea_dim=config.FEATURE_DIM,
            shrink_thres=config.SHRINK_THRES,
            normalize_memory=config.get('NORMALIZE_MEMORY', True),
            normalize_query=config.get('NORMALIZE_QUERY', True),
            use_shared_mlp=True
        )
    else:
        return TemporalMemoryModule(
            feature_dim=config.FEATURE_DIM,
            memory_size=config.TEMPORAL_DIM,
            memory_dim=config.get('MEMORY_DIM', 256),
            shrink_thres=config.SHRINK_THRES
        )


# Test function
if __name__ == "__main__":
    # Test temporal memory module
    config = type('Config', (), {
        'FEATURE_DIM': 512,
        'TEMPORAL_DIM': 2000,
        'MEMORY_DIM': 256,
        'SHRINK_THRES': 0.0025,
        'USE_SHARED_MLP': True,
        'NORMALIZE_MEMORY': True,
        'NORMALIZE_QUERY': True
    })()
    
    # Test both versions
    temp_mem1 = TemporalMemoryModule(512, 2000, 256)
    temp_mem2 = TemporalMemoryWithSharedMLP(2000, 512)
    
    # Test input
    x = torch.randn(2, 512, 32, 32)
    
    # Test forward pass
    with torch.no_grad():
        output1 = temp_mem1(x)
        output2 = temp_mem2(x)
        
        print(f"TemporalMemoryModule output shape: {output1['output'].shape}")
        print(f"TemporalMemoryWithSharedMLP output shape: {output2['output'].shape}")
        print(f"Attention shape: {output1['attention'].shape}")
        
    print("Temporal memory modules test completed!")