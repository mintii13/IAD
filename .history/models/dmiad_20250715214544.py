"""
DMIAD Main Model - Dual Memory Image Anomaly Detection
Tích hợp backbone, dual memory modules, và decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.wider_resnet import build_wideresnet_backbone
from models.memory.temporal_memory import build_temporal_memory
from models.memory.spatial_memory import build_spatial_memory


class ChannelAttention(nn.Module):
    """Channel Attention Module từ code gốc"""
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


class ConvBnRelu(nn.Module):
    """Convolution + BatchNorm + ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvTransposeBnRelu(nn.Module):
    """Transpose Convolution + BatchNorm + ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(ConvTransposeBnRelu, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv_t(x)))


def initialize_weights(*models):
    """Initialize model weights"""
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class FusionModule(nn.Module):
    """Fusion module cho dual memory features"""
    def __init__(self, feature_dim, fusion_method='add'):
        super(FusionModule, self).__init__()
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            self.fusion_conv = nn.Conv2d(feature_dim * 2, feature_dim, 1, bias=False)
        elif fusion_method == 'weighted':
            self.weight_temp = nn.Parameter(torch.ones(1))
            self.weight_spa = nn.Parameter(torch.ones(1))
            
    def forward(self, z_temp, z_spa):
        if self.fusion_method == 'add':
            return z_temp + z_spa
        elif self.fusion_method == 'concat':
            fused = torch.cat([z_temp, z_spa], dim=1)
            return self.fusion_conv(fused)
        elif self.fusion_method == 'weighted':
            weights = F.softmax(torch.stack([self.weight_temp, self.weight_spa]), dim=0)
            return weights[0] * z_temp + weights[1] * z_spa
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class DMIAD(nn.Module):
    def __init__(self, config):
        super(DMIAD, self).__init__()
        
        self.config = config
        self.model_name = config.MODEL.NAME
        
        # Memory configuration - SỬA LỖI: thêm FEATURE_DIM
        self.use_spatial_memory = config.MODEL.MEMORY.USE_SPATIAL
        self.fusion_method = config.MODEL.MEMORY.FUSION_METHOD
        self.feature_dim = getattr(config.MODEL.MEMORY, 'FEATURE_DIM', 512)  # Thêm default value
        
        # === 1. Backbone Network ===
        self.backbone = build_wideresnet_backbone(
            pretrained=config.MODEL.PRETRAINED,
            pretrained_path=getattr(config.MODEL, 'PRETRAINED_PATH', None)
        )
        
        # === 2. Feature Projection Layers ===
        self.conv_proj = nn.Conv2d(4096, self.feature_dim, kernel_size=1, bias=False)
        
        # === 3. Dual Memory Modules ===
        # Temporal memory configuration - SỬA LỖI
        temp_config = type('TempConfig', (), {
            'FEATURE_DIM': self.feature_dim,
            'TEMPORAL_DIM': config.MODEL.MEMORY.TEMPORAL_DIM,
            'MEMORY_DIM': getattr(config.MODEL.MEMORY, 'MEMORY_DIM', 256),
            'SHRINK_THRES': config.MODEL.MEMORY.SHRINK_THRES,
            'USE_SHARED_MLP': getattr(config.MODEL.MEMORY, 'USE_SHARED_MLP', False),
            'NORMALIZE_MEMORY': getattr(config.MODEL.MEMORY, 'NORMALIZE_MEMORY', True),
            'NORMALIZE_QUERY': getattr(config.MODEL.MEMORY, 'NORMALIZE_QUERY', True)
        })()
        
        self.temporal_memory = build_temporal_memory(temp_config)
        
        # Spatial memory configuration - SỬA LỖI
        if self.use_spatial_memory:
            spa_config = type('SpaConfig', (), {
                'FEATURE_DIM': self.feature_dim,
                'SPATIAL_DIM': config.MODEL.MEMORY.SPATIAL_DIM,
                'MEMORY_DIM': getattr(config.MODEL.MEMORY, 'MEMORY_DIM', 256),
                'SHRINK_THRES': config.MODEL.MEMORY.SHRINK_THRES,
                'USE_CHANNEL_WISE': getattr(config.MODEL.MEMORY, 'USE_CHANNEL_WISE', True),
                'USE_POS_ENCODING': getattr(config.MODEL.MEMORY, 'USE_POS_ENCODING', True),
                'NORMALIZE_MEMORY': getattr(config.MODEL.MEMORY, 'NORMALIZE_MEMORY', True),
                'NORMALIZE_QUERY': getattr(config.MODEL.MEMORY, 'NORMALIZE_QUERY', True),
                'USE_SHARED_MLP': getattr(config.MODEL.MEMORY, 'USE_SHARED_MLP', False)
            })()
            
            self.spatial_memory = build_spatial_memory(spa_config)
        
        # === 4. Memory Fusion Module ===
        if self.use_spatial_memory:
            self.fusion = FusionModule(self.feature_dim, self.fusion_method)
        
        # === 5. Attention Modules cho multi-scale features ===
        self.attn_s8 = ChannelAttention(self.feature_dim)
        self.attn_s4 = ChannelAttention(256)
        self.attn_s2 = ChannelAttention(128)
        
        # === 6. Decoder Network ===
        # Upsample từ s8 -> s4
        self.up_s8_to_s4 = ConvTransposeBnRelu(self.feature_dim, 256, kernel_size=2, stride=2)
        
        # Upsample từ s4 -> s2  
        self.up_s4_to_s2 = ConvTransposeBnRelu(256, 128, kernel_size=2, stride=2)
        
        # Upsample từ s2 -> s1
        self.up_s2_to_s1 = ConvTransposeBnRelu(128, 64, kernel_size=2, stride=2)
        
        # === 7. Final Output Layer ===
        self.final_layer = nn.Sequential(
            ConvBnRelu(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        modules_to_init = [
            self.conv_proj, self.conv_s4, self.conv_s2,
            self.up_s8_to_s4, self.up_s4_to_s2, self.up_s2_to_s1,
            self.final_layer
        ]
        
        if self.use_spatial_memory and self.fusion_method == 'concat':
            modules_to_init.append(self.fusion.fusion_conv)
            
        initialize_weights(*modules_to_init)
    
    def extract_features(self, x):
        """Extract multi-scale features using WideResNet backbone"""
        features = self.backbone(x)
        
        return {
            's2': features['s2'],   # [N, 128, H/2, W/2]
            's4': features['s4'],   # [N, 256, H/4, W/4]
            's8': features['s8'],   # [N, 4096, H/8, W/8]
        }
    
    def dual_memory_processing(self, features):
        """Process features through dual memory modules"""
        
        # Project s8 features to efficient dimension
        x = self.conv_proj(features)  # [B, feature_dim, H/8, W/8]
        
        # Store original features for spatial memory
        x_original = x.clone()
        
        # === 1. Temporal Memory Processing ===
        temporal_output = self.temporal_memory(x)
        z_temp = temporal_output['output']  # [B, feature_dim, H/8, W/8]
        temporal_attention = temporal_output.get('attention', None)
        
        # === 2. Spatial Memory Processing (if enabled) ===
        if self.use_spatial_memory:
            spatial_output = self.spatial_memory(x_original)
            z_spa = spatial_output['output']  # [B, feature_dim, H/8, W/8]
            spatial_attention = spatial_output.get('attention', None)
            
            # === 3. Memory Fusion ===
            z_fused = self.fusion(z_temp, z_spa)
        else:
            z_spa = None
            spatial_attention = None
            spatial_output = None
            z_fused = z_temp
        
        return {
            'fused_features': z_fused,
            'temporal_features': z_temp,
            'spatial_features': z_spa,
            'temporal_attention': temporal_attention,
            'spatial_attention': spatial_attention,
            'temporal_output': temporal_output,
            'spatial_output': spatial_output,
        }
    
    def decode_features(self, memory_features, skip_features):
        """Decode features với skip connections và attention"""
        
        x = memory_features  # [B, feature_dim, H/8, W/8]
        
        # Apply attention to memory features
        x = self.attn_s8(x)
        
        # Decode s8 -> s4 với skip connection
        x = self.up_s8_to_s4(x)  # [B, 256, H/4, W/4]
        s4_proj = self.conv_s4(skip_features['s4'])  # Project s4 features
        x = x + s4_proj  # Skip connection
        x = self.attn_s4(x)
        
        # Decode s4 -> s2 với skip connection
        x = self.up_s4_to_s2(x)  # [B, 128, H/2, W/2]
        s2_proj = self.conv_s2(skip_features['s2'])  # Project s2 features
        x = x + s2_proj  # Skip connection
        x = self.attn_s2(x)
        
        # Decode s2 -> s1 (original resolution)
        x = self.up_s2_to_s1(x)  # [B, 64, H, W]
        
        # Final reconstruction
        reconstructed = self.final_layer(x)  # [B, 3, H, W]
        
        return reconstructed
    
    def forward(self, x):
        """Forward pass"""
        
        # === 1. Feature Extraction ===
        feature_dict = self.extract_features(x)
        
        # === 2. Dual Memory Processing ===
        memory_results = self.dual_memory_processing(feature_dict['s8'])
        
        # === 3. Decoding với Skip Connections ===
        reconstructed = self.decode_features(
            memory_results['fused_features'],
            feature_dict
        )
        
        # === 4. Prepare Output Dictionary ===
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
        
        # Image-level anomaly score
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
            memory_items['temporal_memory'] = self.temporal_memory.memory
        elif hasattr(self.temporal_memory, 'C'):
            memory_items['temporal_memory'] = self.temporal_memory.C
        
        # Spatial memory items
        if self.use_spatial_memory:
            if hasattr(self.spatial_memory, 'memory') and hasattr(self.spatial_memory.memory, 'weight'):
                memory_items['spatial_memory'] = self.spatial_memory.memory.weight.data
            elif hasattr(self.spatial_memory, 'H'):
                memory_items['spatial_memory'] = self.spatial_memory.H
                
        return memory_items


def build_dmiad_model(config):
    """Build DMIAD model based on configuration"""
    return DMIAD(config)


# Example usage và testing
if __name__ == "__main__":
    # Test model creation
    from types import SimpleNamespace
    
    # Create dummy config
    config = SimpleNamespace()
    config.MODEL = SimpleNamespace()
    config.MODEL.NAME = 'dmiad'
    config.MODEL.PRETRAINED = False
    config.MODEL.PRETRAINED_PATH = None
    
    config.MODEL.MEMORY = SimpleNamespace()
    config.MODEL.MEMORY.USE_SPATIAL = True
    config.MODEL.MEMORY.FUSION_METHOD = 'add'
    config.MODEL.MEMORY.FEATURE_DIM = 512
    config.MODEL.MEMORY.TEMPORAL_DIM = 2000
    config.MODEL.MEMORY.SPATIAL_DIM = 2000
    config.MODEL.MEMORY.MEMORY_DIM = 256
    config.MODEL.MEMORY.SHRINK_THRES = 0.0025
    config.MODEL.MEMORY.USE_SHARED_MLP = False
    config.MODEL.MEMORY.USE_CHANNEL_WISE = True
    config.MODEL.MEMORY.USE_POS_ENCODING = True
    config.MODEL.MEMORY.NORMALIZE_MEMORY = True
    config.MODEL.MEMORY.NORMALIZE_QUERY = True
    
    # Build model
    model = build_dmiad_model(config)
    print(f"Model: {model.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Reconstructed shape: {output['reconstructed'].shape}")
        
        # Test anomaly scoring
        scores = model.compute_anomaly_score(dummy_input, output['reconstructed'])
        print(f"Image scores shape: {scores['image_scores'].shape}")
        print(f"Pixel scores shape: {scores['pixel_scores'].shape}")
        
        # Test memory items
        memory_items = model.get_memory_items()
        for name, item in memory_items.items():
            print(f"{name} shape: {item.shape}")
    
    print("DMIAD model test completed successfully!")