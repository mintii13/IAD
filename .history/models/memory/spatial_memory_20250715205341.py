"""
Spatial Memory Module - theo pipeline diagram
Xử lý spatial patterns với SSIM-like similarity và channel-wise processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


class MLPModule(nn.Module):
    """MLP module for spatial memory processing như trong pipeline"""
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


class PositionalEncodingChannelWise(nn.Module):
    """Channel-wise Positional Encoding cho spatial patterns"""
    def __init__(self, spatial_dim, max_height=64, max_width=64):
        super(PositionalEncodingChannelWise, self).__init__()
        self.spatial_dim = spatial_dim
        self.max_height = max_height
        self.max_width = max_width
        
        # Pre-compute positional encodings
        max_spatial_dim = max_height * max_width
        pos_embed = torch.zeros(max_spatial_dim)
        
        for h in range(max_height):
            for w in range(max_width):
                spatial_idx = h * max_width + w
                pos_embed[spatial_idx] = torch.sin(torch.tensor(spatial_idx / (10000 ** (spatial_idx / max_spatial_dim))))
        
        self.register_buffer('pos_embed', pos_embed)
    
    def forward(self, x_flat, height, width):
        """Add positional encoding to flattened spatial patterns"""
        current_spatial_dim = height * width
        device = x_flat.device
        
        if height <= self.max_height and width <= self.max_width:
            pe_indices = []
            for h in range(height):
                for w in range(width):
                    pe_indices.append(h * self.max_width + w)
            
            pe = self.pos_embed[pe_indices]
            pe = pe.unsqueeze(0).expand(x_flat.shape[0], -1)
        else:
            pe = torch.zeros_like(x_flat)
            for h in range(height):
                for w in range(width):
                    spatial_idx = h * width + w
                    pe_value = torch.sin(torch.tensor(spatial_idx / (10000 ** (spatial_idx / current_spatial_dim)), 
                                                    device=device, dtype=x_flat.dtype))
                    pe[:, spatial_idx] = pe_value
        
        return x_flat + pe * 0.1


class SpatialMemoryModule(nn.Module):
    """
    Spatial Memory Module theo pipeline diagram
    
    Pipeline: Z -> MLP_spa -> Normalize -> SSIM(Z_spa, M_spa) -> W_spa -> Memsize -> Z_spa
    Sử dụng SSIM similarity thay vì simple dot product
    """
    def __init__(self, feature_dim, memory_size, memory_dim, shrink_thres=0.0025):
        super(SpatialMemoryModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.shrink_thres = shrink_thres
        
        # Spatial memory bank H (như trong diagram) 
        # Organize như spatial structure
        memory_h = int(math.sqrt(memory_size))
        memory_w = memory_size // memory_h
        self.memory_h = memory_h
        self.memory_w = memory_w
        
        self.register_buffer('H', torch.randn(memory_h, memory_w, memory_dim))
        self.H = F.normalize(self.H, dim=-1)
        
        # MLP modules với shared weights (như trong diagram)
        self.mlp_spa_query = MLPModule(feature_dim, feature_dim//2, memory_dim, normalize=True)
        self.mlp_spa_memory = MLPModule(memory_dim, memory_dim//2, memory_dim, normalize=True)
        
        # Memsize operation
        self.memsize_spa = nn.Linear(memory_dim, feature_dim)
        
    def compute_ssim_similarity(self, Z_spa, M_spa_processed):
        """
        Compute SSIM-like similarity cho spatial patterns
        SSIM(Z_spa, M_spa) trong pipeline diagram
        """
        B, H, W, D = Z_spa.shape
        
        # Flatten cho similarity computation
        Z_flat = Z_spa.view(B*H*W, D)  # [B*H*W, D]
        M_flat = M_spa_processed.view(self.memory_h*self.memory_w, D)  # [mH*mW, D]
        
        # Compute mean
        mu_z = torch.mean(Z_flat, dim=1, keepdim=True)  # [B*H*W, 1]
        mu_m = torch.mean(M_flat, dim=1, keepdim=True)  # [mH*mW, 1]
        
        # Compute variance
        var_z = torch.var(Z_flat, dim=1, keepdim=True)  # [B*H*W, 1]
        var_m = torch.var(M_flat, dim=1, keepdim=True)  # [mH*mW, 1]
        
        # Compute covariance (simplified SSIM-like)
        Z_centered = Z_flat - mu_z
        M_centered = M_flat - mu_m
        
        # SSIM-like similarity computation
        # (2*mu_z*mu_m + c1) * (2*cov + c2) / ((mu_z^2 + mu_m^2 + c1) * (var_z + var_m + c2))
        c1, c2 = 0.01, 0.03
        
        # Simplified version: use cosine similarity với variance weighting
        similarity = torch.matmul(F.normalize(Z_centered, dim=1), 
                                F.normalize(M_centered, dim=1).t())
        
        # Weight by variance similarity
        var_similarity = 1.0 / (1.0 + torch.abs(var_z - var_m.t()))
        similarity = similarity * var_similarity
        
        return similarity  # [B*H*W, mH*mW]
    
    def forward(self, z):
        """
        Spatial memory processing theo pipeline diagram
        
        Args:
            z: Input features [B, C, H, W]
        Returns:
            dict với output và attention information
        """
        B, C, H, W = z.shape
        
        # Channel-wise flatten cho spatial processing
        # [B, C, H, W] -> [B*C, H*W] - mỗi row là 1 channel's spatial pattern
        z_spatial = z.view(B * C, H * W)
        
        # Reshape for MLP processing: [B*C, H*W] -> [B, C, H*W] -> [B, H*W, C]
        z_for_mlp = z_spatial.view(B, C, H*W).permute(0, 2, 1).contiguous().view(B*H*W, C)
        
        # === PIPELINE STEP 1: Process query qua MLP ===
        Z_spa = self.mlp_spa_query(z_for_mlp)  # [B*H*W, memory_dim]
        Z_spa = Z_spa.view(B, H, W, self.memory_dim)  # [B, H, W, memory_dim]
        
        # === PIPELINE STEP 2: Process memory qua MLP ===  
        M_spa = self.mlp_spa_memory(self.H)  # [memory_h, memory_w, memory_dim]
        
        # === PIPELINE STEP 3: Compute SSIM similarity ===
        # SSIM(Z_spa, M_spa) trong diagram
        similarity = self.compute_ssim_similarity(Z_spa, M_spa)  # [B*H*W, memory_h*memory_w]
        
        # === PIPELINE STEP 4: Compute attention weights ===
        W_spa = F.softmax(similarity, dim=1)  # [B*H*W, memory_h*memory_w]
        
        # Shrinkage operation (optional)
        if self.shrink_thres > 0:
            W_spa = F.relu(W_spa - self.shrink_thres)
            W_spa = F.normalize(W_spa, p=1, dim=1)
        
        # === PIPELINE STEP 5: Memory retrieval ===
        # W_spa × M_spa
        M_flat = M_spa.view(-1, self.memory_dim)  # [memory_h*memory_w, memory_dim]
        retrieved_memory = torch.matmul(W_spa, M_flat)  # [B*H*W, memory_dim]
        
        # === PIPELINE STEP 6: Memsize operation ===
        # Transform memory về feature space
        Z_spa_out = self.memsize_spa(retrieved_memory)  # [B*H*W, feature_dim]
        
        # === PIPELINE STEP 7: Reshape về spatial format ===
        Z_spa_final = Z_spa_out.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Convert attention format for compatibility
        att_reshaped = W_spa.view(B, H, W, self.memory_h * self.memory_w)
        att_spatial = att_reshaped.mean(dim=(1, 2))  # [B, memory_h*memory_w]
        att_spatial = att_spatial.unsqueeze(-1).unsqueeze(-1).expand(B, self.memory_h * self.memory_w, H, W)
        
        return {
            'output': Z_spa_final,  # Reconstructed features
            'attention': att_spatial,  # [B, memory_size, H, W] - compatible format
            'retrieved_memory': retrieved_memory.view(B, H, W, self.memory_dim),
            'processed_query': Z_spa,  # [B, H, W, memory_dim]
            'processed_memory': M_spa,  # Processed memory bank
            'raw_attention': W_spa  # Raw attention [B*H*W, memory_size]
        }


class SpatialMemoryWithChannelWise(nn.Module):
    """
    Spatial Memory với channel-wise processing
    Symmetric với temporal memory's pixel-wise processing
    """
    def __init__(self, mem_dim, shrink_thres=0.0025, use_pos_encoding=True,
                 normalize_memory=True, normalize_query=True, use_shared_mlp=False):
        super(SpatialMemoryWithChannelWise, self).__init__()
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.use_pos_encoding = use_pos_encoding
        self.normalize_memory = normalize_memory
        self.normalize_query = normalize_query
        self.use_shared_mlp = use_shared_mlp
        
        # Dynamic initialization based on input
        self.spatial_dim = None
        self.memory = None
        self.pos_encoding = None
        
        # Dummy parameter for device detection
        self.register_parameter('_dummy', Parameter(torch.tensor(0.0)))

    def _initialize_modules(self, height, width):
        """Initialize memory unit và PE based on input spatial dimensions"""
        if self.spatial_dim is None:
            self.spatial_dim = height * width
            device = self._dummy.device
            
            # Initialize channel-wise memory unit
            self.memory = MemoryUnitChannelWise(
                self.mem_dim, self.spatial_dim, self.shrink_thres,
                normalize_memory=self.normalize_memory, 
                normalize_query=self.normalize_query,
                use_shared_mlp=self.use_shared_mlp
            ).to(device)
            
            # Initialize positional encoding
            if self.use_pos_encoding:
                self.pos_encoding = PositionalEncodingChannelWise(
                    self.spatial_dim, height, width
                ).to(device)

    def forward(self, input):
        """
        Channel-wise spatial memory processing
        Symmetric với temporal memory's pixel-wise processing
        """
        s = input.data.shape
        N, C, H, W = s
        
        # Initialize modules on first forward pass
        self._initialize_modules(H, W)
        
        # Channel-wise flatten: [N,C,H,W] -> [N*C,H*W]
        x = input.view(N * C, H * W)  # Each row is one channel's spatial pattern
        
        # Add positional encoding
        if self.use_pos_encoding:
            x = self.pos_encoding(x, height=H, width=W)
        
        # Apply channel-wise memory unit
        y_and = self.memory(x)
        y = y_and['output']      # [N*C, H*W]
        att = y_and['att']       # [N*C, M]
        mem_fea_align = y_and['mem_fea_align']
        processed_memory = y_and['processed_memory']

        # Reshape back: [N*C, H*W] -> [N,C,H,W]
        y = y.view(N, C, H, W)
        
        # Convert attention format for compatibility
        att_reshaped = att.view(N, C, self.mem_dim)
        att_channel_avg = att_reshaped.mean(dim=1)  # [N, M]
        att_spatial = att_channel_avg.unsqueeze(-1).unsqueeze(-1).expand(N, self.mem_dim, H, W)

        return {
            'output': y,
            'att': att_spatial,
            'mem_fea_align': mem_fea_align,
            'processed_memory': processed_memory,
            'input_flat': x,
            'raw_attention': att
        }


class MemoryUnitChannelWise(nn.Module):
    """Memory unit cho channel-wise processing"""
    def __init__(self, mem_dim, spatial_dim, shrink_thres=0.005, 
                 normalize_memory=True, normalize_query=True, use_shared_mlp=False):
        super(MemoryUnitChannelWise, self).__init__()
        self.mem_dim = mem_dim
        self.spatial_dim = spatial_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.spatial_dim))
        self.shrink_thres = shrink_thres
        self.normalize_memory = normalize_memory
        self.normalize_query = normalize_query
        self.use_shared_mlp = use_shared_mlp
        
        if self.use_shared_mlp:
            self.shared_mlp = nn.Sequential(
                nn.Linear(spatial_dim, spatial_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(spatial_dim // 2, spatial_dim),
                nn.ReLU(inplace=True)
            )
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """Channel-wise memory processing"""
        if self.normalize_query:
            input_norm = F.normalize(input, dim=1)
        else:
            input_norm = input
            
        if self.normalize_memory:
            weight_norm = F.normalize(self.weight, dim=1)
        else:
            weight_norm = self.weight
        
        if self.use_shared_mlp:
            processed_memory = self.shared_mlp(weight_norm)
            if self.normalize_memory:
                processed_memory = F.normalize(processed_memory, dim=1)
        else:
            processed_memory = weight_norm
        
        att_weight = F.linear(input_norm, processed_memory)
        mem_fea_align = F.linear(processed_memory, processed_memory)
        att_weight = F.softmax(att_weight, dim=1)
        
        mem_trans = processed_memory.permute(1, 0)
        output = F.linear(att_weight, mem_trans)
        
        return {
            'output': output,
            'att': att_weight,
            'mem_fea_align': mem_fea_align,
            'processed_memory': processed_memory
        }


def build_spatial_memory(config):
    """
    Build spatial memory module based on config
    """
    if hasattr(config, 'USE_CHANNEL_WISE') and config.USE_CHANNEL_WISE:
        return SpatialMemoryWithChannelWise(
            mem_dim=config.SPATIAL_DIM,
            shrink_thres=config.SHRINK_THRES,
            use_pos_encoding=config.get('USE_POS_ENCODING', True),
            normalize_memory=config.get('NORMALIZE_MEMORY', True),
            normalize_query=config.get('NORMALIZE_QUERY', True),
            use_shared_mlp=config.get('USE_SHARED_MLP', False)
        )
    else:
        return SpatialMemoryModule(
            feature_dim=config.FEATURE_DIM,
            memory_size=config.SPATIAL_DIM,
            memory_dim=config.get('MEMORY_DIM', 256),
            shrink_thres=config.SHRINK_THRES
        )


# # Test function
# if __name__ == "__main__":
#     # Test spatial memory module
#     config = type('Config', (), {
#         'FEATURE_DIM': 512,
#         'SPATIAL_DIM': 2000,
#         'MEMORY_DIM': 256,
#         'SHRINK_THRES': 0.0025,
#         'USE_CHANNEL_WISE': True,
#         'USE_POS_ENCODING': True,
#         'NORMALIZE_MEMORY': True,
#         'NORMALIZE_QUERY': True,
#         'USE_SHARED_MLP': True
#     })()
    
#     # Test both versions
#     spa_mem1 = SpatialMemoryModule(512, 2000, 256)
#     spa_mem2 = SpatialMemoryWithChannelWise(2000)
    
#     # Test input
#     x = torch.randn(2, 512, 32, 32)
    
#     # Test forward pass
#     with torch.no_grad():
#         output1 = spa_mem1(x)
#         output2 = spa_mem2(x)
        
#         print(f"SpatialMemoryModule output shape: {output1['output'].shape}")
#         print(f"SpatialMemoryWithChannelWise output shape: {output2['output'].shape}")
#         print(f"Attention shape: {output1['attention'].shape}")
        
#     print("Spatial memory modules test completed!")