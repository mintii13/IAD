"""
DMIAD Model - Dual Memory Image Anomaly Detection
Combines temporal and spatial memory modules for image anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.wide_resnet import WideResNet  # From CRAS
from models.memory.temporal_memory import MemModule  # From CRAS/Video model
from models.memory.spatial_memory import SpatialMemModule  # New spatial memory
from models.modules.basic_modules import ConvBnRelu, ConvTransposeBnRelu, initialize_weights


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, input_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        reduced_channels = max(input_channels // reduction, 8)
        
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, input_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.layer(y)
        return x * y


class DMIAD(nn.Module):
    """
    Dual Memory Image Anomaly Detection Model
    
    Architecture:
    1. Backbone feature extraction (WideResNet)
    2. Dual Memory Processing:
       - Temporal Memory: Process feature channels (adapted from video model)
       - Spatial Memory: Process spatial patterns 
    3. Memory Fusion
    4. Reconstruction Decoder
    """
    
    def __init__(self, config):
        super(DMIAD, self).__init__()
        
        self.config = config
        self.model_name = config.MODEL.NAME
        
        # Memory configuration
        self.use_spatial_memory = config.MODEL.MEMORY.USE_SPATIAL
        self.fusion_method = config.MODEL.MEMORY.FUSION_METHOD
        self.temporal_mem_dim = config.MODEL.MEMORY.TEMPORAL_DIM
        self.spatial_mem_dim = config.MODEL.MEMORY.SPATIAL_DIM
        
        # Feature channels at different scales
        # These match WideResNet architecture from CRAS
        self.feature_channels = {
            'layer1': 128,   # Early features
            'layer2': 256,   # Mid features  
            'layer3': 512,   # High-level features
            'layer4': 1024,  # Deep features
        }
        
        # 1. Backbone Network (Feature Extractor)
        self.backbone = self._build_backbone()
        
        # 2. Feature Projection Layers
        # Project concatenated features to efficient dimensions
        self.feature_proj = nn.Conv2d(
            self.feature_channels['layer4'], 
            512,  # Efficient channel dimension
            kernel_size=1, 
            bias=False
        )
        
        # 3. Dual Memory Modules
        self.temporal_memory = MemModule(
            mem_dim=self.temporal_mem_dim,
            fea_dim=512,  # Projected feature dimension
            shrink_thres=config.MODEL.MEMORY.SHRINK_THRES
        )
        
        if self.use_spatial_memory:
            self.spatial_memory = SpatialMemModule(
                mem_dim=self.spatial_mem_dim,
                shrink_thres=config.MODEL.MEMORY.SHRINK_THRES,
                use_pos_encoding=True,
                normalize_memory=config.MODEL.MEMORY.NORMALIZE_MEMORY,
                normalize_query=config.MODEL.MEMORY.NORMALIZE_QUERY,
                use_shared_mlp=config.MODEL.MEMORY.USE_SHARED_MLP
            )
        
        # 4. Memory Fusion Layer (if using concat fusion)
        if self.use_spatial_memory and self.fusion_method == 'concat':
            self.fusion_conv = nn.Conv2d(
                512 * 2,  # Concatenated temporal + spatial
                512,      # Fused features
                kernel_size=1,
                bias=False
            )
        
        # 5. Attention Modules for multi-scale features
        self.attn_modules = nn.ModuleDict({
            'attn1': ChannelAttention(256),
            'attn2': ChannelAttention(128), 
            'attn3': ChannelAttention(64),
        })
        
        # 6. Decoder Network (Reconstruction)
        self.decoder = self._build_decoder()
        
        # 7. Final Output Layer
        self.final_layer = nn.Sequential(
            ConvBnRelu(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_backbone(self):
        """Build feature extraction backbone"""
        # TODO: Implement WideResNet backbone from CRAS
        # This should extract multi-scale features
        # Return a backbone that outputs features at different scales
        
        backbone = WideResNet(
            layers=[3, 4, 6, 3],  # WideResNet architecture
            pretrained=self.config.MODEL.PRETRAINED
        )
        
        return backbone
    
    def _build_decoder(self):
        """Build reconstruction decoder"""
        decoder = nn.Sequential(
            # Upsample from 512 -> 256 channels
            ConvTransposeBnRelu(512, 256, kernel_size=2, stride=2),
            
            # Upsample from 256 -> 128 channels  
            ConvTransposeBnRelu(256, 128, kernel_size=2, stride=2),
            
            # Upsample from 128 -> 64 channels
            ConvTransposeBnRelu(128, 64, kernel_size=2, stride=2),
        )
        
        return decoder
    
    def _initialize_weights(self):
        """Initialize model weights"""
        modules_to_init = [
            self.feature_proj, self.decoder, self.final_layer
        ]
        
        if hasattr(self, 'fusion_conv'):
            modules_to_init.append(self.fusion_conv)
            
        initialize_weights(*modules_to_init)
    
    def extract_features(self, x):
        """Extract multi-scale features using backbone"""
        # TODO: Implement feature extraction
        # Should return features at different scales for skip connections
        
        features = self.backbone(x)
        
        # Extract features at different scales
        # This is a placeholder - implement based on actual backbone
        feat_dict = {
            'layer1': features[0],  # Early features
            'layer2': features[1],  # Mid features
            'layer3': features[2],  # High features  
            'layer4': features[3],  # Deep features
        }
        
        return feat_dict
    
    def dual_memory_processing(self, features):
        """Process features through dual memory modules"""
        
        # Project features to efficient dimension
        x = self.feature_proj(features)  # [B, 512, H, W]
        
        # Store original features for spatial memory
        x_original = x.clone()
        
        # 1. Temporal Memory Processing
        # Process feature channels (adapted from video temporal processing)
        temporal_output = self.temporal_memory(x)
        z_temporal = temporal_output['output']  # [B, 512, H, W]
        temporal_attention = temporal_output['att']  # Attention maps
        
        # 2. Spatial Memory Processing (if enabled)
        if self.use_spatial_memory:
            spatial_output = self.spatial_memory(x_original)
            z_spatial = spatial_output['output']  # [B, 512, H, W]
            spatial_attention = spatial_output['att']  # Attention maps
            
            # 3. Memory Fusion
            z_fused = self.fuse_memory_features(z_temporal, z_spatial)
        else:
            z_spatial = None
            spatial_attention = None
            spatial_output = None
            z_fused = z_temporal
        
        return {
            'fused_features': z_fused,
            'temporal_features': z_temporal,
            'spatial_features': z_spatial,
            'temporal_attention': temporal_attention,
            'spatial_attention': spatial_attention,
            'temporal_output': temporal_output,
            'spatial_output': spatial_output,
        }
    
    def fuse_memory_features(self, z_temporal, z_spatial):
        """Fuse temporal and spatial memory features"""
        
        if self.fusion_method == 'add':
            z_fused = z_temporal + z_spatial
        elif self.fusion_method == 'avg':
            z_fused = (z_temporal + z_spatial) / 2.0
        elif self.fusion_method == 'concat':
            z_concat = torch.cat([z_temporal, z_spatial], dim=1)
            z_fused = self.fusion_conv(z_concat)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return z_fused
    
    def reconstruct_image(self, memory_features, skip_features):
        """Reconstruct image using decoder with skip connections"""
        
        x = memory_features  # [B, 512, H/8, W/8]
        
        # Decoder with skip connections and attention
        x = self.decoder[0](x)  # -> [B, 256, H/4, W/4]
        if 'layer2' in skip_features:
            x = x + skip_features['layer2']  # Skip connection
        x = self.attn_modules['attn1'](x)
        
        x = self.decoder[1](x)  # -> [B, 128, H/2, W/2]
        if 'layer1' in skip_features:
            x = x + skip_features['layer1']  # Skip connection  
        x = self.attn_modules['attn2'](x)
        
        x = self.decoder[2](x)  # -> [B, 64, H, W]
        x = self.attn_modules['attn3'](x)
        
        # Final reconstruction
        reconstructed = self.final_layer(x)  # -> [B, 3, H, W]
        
        return reconstructed
    
    def forward(self, x):
        """Forward pass"""
        
        # 1. Feature Extraction
        feature_dict = self.extract_features(x)
        
        # 2. Dual Memory Processing
        memory_results = self.dual_memory_processing(feature_dict['layer4'])
        
        # 3. Image Reconstruction 
        reconstructed = self.reconstruct_image(
            memory_results['fused_features'],
            feature_dict
        )
        
        # 4. Prepare output dictionary
        output_dict = {
            'reconstructed': reconstructed,
            'memory_results': memory_results,
            'features': feature_dict,
        }
        
        return output_dict
    
    def compute_anomaly_score(self, x, reconstructed):
        """Compute anomaly score based on reconstruction error"""
        
        # Pixel-wise reconstruction error
        pixel_error = torch.mean((x - reconstructed) ** 2, dim=1, keepdim=True)
        
        # Image-level anomaly score (average over spatial dimensions)
        image_score = torch.mean(pixel_error, dim=[2, 3])
        
        return {
            'pixel_scores': pixel_error,  # [B, 1, H, W]
            'image_scores': image_score,  # [B, 1]
        }
    
    def get_memory_items(self):
        """Get memory items for analysis"""
        memory_items = {}
        
        # Temporal memory items
        if hasattr(self.temporal_memory, 'memory'):
            memory_items['temporal_memory'] = self.temporal_memory.memory.weight.data
        
        # Spatial memory items  
        if self.use_spatial_memory and hasattr(self.spatial_memory, 'spatial_memory'):
            memory_items['spatial_memory'] = self.spatial_memory.spatial_memory.weight.data
            
        return memory_items


class DensityEstimationHead(nn.Module):
    """Optional density estimation head for advanced anomaly scoring"""
    
    def __init__(self, feature_dim=512, hidden_dim=256):
        super(DensityEstimationHead, self).__init__()
        
        self.density_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """Estimate density/normality score"""
        density_score = self.density_net(features)
        return density_score


class DMIADWithDensity(DMIAD):
    """DMIAD model with additional density estimation"""
    
    def __init__(self, config):
        super(DMIADWithDensity, self).__init__(config)
        
        # Add density estimation head
        self.density_head = DensityEstimationHead(feature_dim=512)
    
    def forward(self, x):
        """Forward pass with density estimation"""
        
        # Get base DMIAD outputs
        output_dict = super().forward(x)
        
        # Compute density score
        memory_features = output_dict['memory_results']['fused_features']
        density_score = self.density_head(memory_features)
        
        output_dict['density_score'] = density_score
        
        return output_dict


def build_dmiad_model(config):
    """Build DMIAD model based on config"""
    
    if config.MODEL.get('USE_DENSITY', False):
        model = DMIADWithDensity(config)
    else:
        model = DMIAD(config)
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    from config.base_config import get_config
    import argparse
    
    # Create dummy config
    args = argparse.Namespace()
    args.dataset = 'mvtec'
    args.data_path = '/path/to/data'
    args.class_name = 'bottle'
    args.setting = 'single'
    args.batch_size = 4
    args.gpu = 0
    args.mem_dim = 1000
    args.use_spatial_memory = True
    args.fusion_method = 'add'
    args.backbone = 'wide_resnet'
    args.epochs = 100
    args.lr = 1e-4
    args.output_dir = './results'
    args.exp_name = 'test'
    
    config = get_config(args)
    
    # Build model
    model = build_dmiad_model(config)
    print(f"Model: {model.__class__.__name__}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Reconstructed shape: {output['reconstructed'].shape}")
        
        # Test anomaly scoring
        scores = model.compute_anomaly_score(dummy_input, output['reconstructed'])
        print(f"Image scores shape: {scores['image_scores'].shape}")
        print(f"Pixel scores shape: {scores['pixel_scores'].shape}")
        
        # Test memory items
        memory_items = model.get_memory_items()
        for name, item in memory_items.items():
            print(f"{name} shape: {item.shape}")
    
    print("Model test completed successfully!")