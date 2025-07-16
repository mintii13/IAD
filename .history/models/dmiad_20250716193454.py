"""
DMIAD Main Model - Updated with MobileNet Support
Supports both WideResNet and MobileNet backbones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.wide_resnet import build_wideresnet_backbone
from models.backbone.mobilenet import build_mobilenet_backbone, MobileNetDecoder
from models.memory.temporal_memory import build_temporal_memory
from models.memory.spatial_memory import build_spatial_memory


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


class FusionModule(nn.Module):
    """Fusion module for dual memory features"""
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
        
        # Backbone configuration
        self.backbone_type = getattr(config.MODEL, 'BACKBONE', 'wide_resnet')
        
        # Memory configuration
        self.use_spatial_memory = config.MODEL.MEMORY.USE_SPATIAL
        self.fusion_method = config.MODEL.MEMORY.FUSION_METHOD
        
        # === 1. Backbone Network ===
        self._build_backbone(config)
        
        # === 2. Auto-detect feature dimensions from backbone ===
        self._detect_backbone_dims(config)
        
        # === 3. Feature Projection Layers ===
        self._build_feature_projections()
        
        # === 4. Dual Memory Modules ===
        self._build_memory_modules(config)
        
        # === 5. Memory Fusion Module ===
        if self.use_spatial_memory:
            self.fusion = FusionModule(self.feature_dim, self.fusion_method)
        
        # === 6. Decoder Network ===
        self._build_decoder()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_backbone(self, config):
        """Build backbone network based on configuration"""
        
        if self.backbone_type.startswith('mobilenet'):
            # Parse MobileNet configuration
            if 'v2' in self.backbone_type:
                version = 'v2'
            elif 'v3' in self.backbone_type:
                version = 'v3'
            else:
                version = 'v2'  # Default
            
            # Get width multiplier from backbone name if specified
            width_mult = 1.0
            if '_' in self.backbone_type:
                parts = self.backbone_type.split('_')
                for part in parts:
                    if part.replace('.', '').isdigit():
                        width_mult = float(part)
                        break
            
            self.backbone = build_mobilenet_backbone(
                version=version,
                pretrained=config.MODEL.PRETRAINED,
                pretrained_path=getattr(config.MODEL, 'PRETRAINED_PATH', None),
                width_mult=width_mult
            )
            
        elif self.backbone_type == 'wide_resnet':
            self.backbone = build_wideresnet_backbone(
                pretrained=config.MODEL.PRETRAINED,
                pretrained_path=getattr(config.MODEL, 'PRETRAINED_PATH', None)
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_type}")
        
        print(f"Using backbone: {self.backbone_type}")
    
    def _detect_backbone_dims(self, config):
        """Auto-detect backbone output dimensions"""
        # Create dummy input to get backbone dimensions
        dummy_input = torch.randn(1, 3, config.DATASET.CROP_SIZE, config.DATASET.CROP_SIZE)
        
        with torch.no_grad():
            features = self.backbone(dummy_input)
            
            # Get s8 feature dimensions (main features for memory processing)
            s8_features = features['s8']  # [1, C, H, W]
            self.backbone_feature_dim = s8_features.shape[1]  # Channel dimension
            self.spatial_h = s8_features.shape[2]  # Height
            self.spatial_w = s8_features.shape[3]  # Width
            
            # Set projected feature dimension based on backbone
            if self.backbone_type.startswith('mobilenet'):
                self.feature_dim = 256  # Smaller for MobileNet
            else:
                self.feature_dim = 512  # Original for WideResNet
            
        print(f"Detected backbone dimensions:")
        print(f"  - Backbone feature dim: {self.backbone_feature_dim}")
        print(f"  - Spatial size: {self.spatial_h}x{self.spatial_w}")
        print(f"  - Projected feature dim: {self.feature_dim}")
    
    def _build_feature_projections(self):
        """Build feature projection layers"""
        # Main projection for s8 features
        self.conv_proj = nn.Conv2d(self.backbone_feature_dim, self.feature_dim, kernel_size=1, bias=False)
        
        # Skip connection projections are handled by the backbone
        # No need for separate projections since backbone already outputs standardized channels
    
    def _build_memory_modules(self, config):
        """Build memory modules with detected dimensions"""
        
        # Temporal memory config
        temp_config = type('TempConfig', (), {
            'FEATURE_DIM': self.feature_dim,
            'TEMPORAL_DIM': config.MODEL.MEMORY.TEMPORAL_DIM,
            'SHRINK_THRES': config.MODEL.MEMORY.SHRINK_THRES,
        })()
        
        self.temporal_memory = build_temporal_memory(temp_config)
        
        # Spatial memory config
        if self.use_spatial_memory:
            spa_config = type('SpaConfig', (), {
                'SPATIAL_DIM': config.MODEL.MEMORY.SPATIAL_DIM,
                'SHRINK_THRES': config.MODEL.MEMORY.SHRINK_THRES,
                'SPATIAL_H': self.spatial_h,
                'SPATIAL_W': self.spatial_w,
            })()
            
            self.spatial_memory = build_spatial_memory(spa_config)
    
    def _build_decoder(self):
        """Build decoder network based on backbone type"""
        
        if self.backbone_type.startswith('mobilenet'):
            # Use lightweight MobileNet decoder
            self.decoder = MobileNetDecoder(input_dim=self.feature_dim)
        else:
            # Use original decoder for WideResNet
            self._build_original_decoder()
    
    def _build_original_decoder(self):
        """Build original decoder for WideResNet backbone"""
        # Attention modules
        self.attn_s8 = ChannelAttention(self.feature_dim)
        self.attn_s4 = ChannelAttention(256)
        self.attn_s2 = ChannelAttention(128)
        
        # Skip connection projections
        self.conv_s4 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.conv_s2 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        
        # Decoder layers
        self.up_s8_to_s4 = ConvTransposeBnRelu(self.feature_dim, 256, kernel_size=2, stride=2)
        self.up_s4_to_s2 = ConvTransposeBnRelu(256, 128, kernel_size=2, stride=2)
        self.up_s2_to_s1 = ConvTransposeBnRelu(128, 64, kernel_size=2, stride=2)
        
        # Final output layer
        self.final_layer = nn.Sequential(
            ConvBnRelu(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False)
        )
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def dual_memory_processing(self, features):
        """Process features through dual memory modules"""
        
        # Project s8 features to efficient dimension
        x = self.conv_proj(features)  # [B, feature_dim, H/8, W/8]
        
        # === 1. Temporal Memory Processing (pixel-wise) ===
        temporal_output = self.temporal_memory(x)
        z_temp = temporal_output['output']  # [B, feature_dim, H/8, W/8]
        temporal_attention = temporal_output.get('att', None)
        
        # === 2. Spatial Memory Processing (channel-wise) ===
        if self.use_spatial_memory:
            spatial_output = self.spatial_memory(x)
            z_spa = spatial_output['output']  # [B, feature_dim, H/8, W/8]
            spatial_attention = spatial_output.get('att', None)
            
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
    
    def extract_features(self, x):
        """Extract multi-scale features using backbone"""
        features = self.backbone(x)
        return features
    
    def decode_features(self, memory_features, skip_features):
        """Decode features with appropriate decoder"""
        
        if self.backbone_type.startswith('mobilenet'):
            # Use MobileNet decoder
            reconstructed = self.decoder(memory_features, skip_features)
        else:
            # Use original decoder for WideResNet
            reconstructed = self._decode_with_original_decoder(memory_features, skip_features)
        
        return reconstructed
    
    def _decode_with_original_decoder(self, memory_features, skip_features):
        """Decode features with original decoder (for WideResNet)"""
        x = memory_features  # [B, feature_dim, H/8, W/8]
        
        # Apply attention to memory features
        x = self.attn_s8(x)
        
        # Decode s8 -> s4 with skip connection
        x = self.up_s8_to_s4(x)  # [B, 256, H/4, W/4]
        s4_proj = self.conv_s4(skip_features['s4'])
        x = x + s4_proj
        x = self.attn_s4(x)
        
        # Decode s4 -> s2 with skip connection
        x = self.up_s4_to_s2(x)  # [B, 128, H/2, W/2]
        s2_proj = self.conv_s2(skip_features['s2'])
        x = x + s2_proj
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
        
        # === 3. Decoding with Skip Connections ===
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
            elif hasattr(self.spatial_memory, 'memory'):
                memory_items['spatial_memory'] = self.spatial_memory.memory
                
        return memory_items


def build_dmiad_model(config):
    """Build DMIAD model based on configuration"""
    return DMIAD(config)


# Test function with MobileNet
def test_dmiad_with_mobilenet():
    """Test DMIAD model with MobileNet backbone"""
    from types import SimpleNamespace
    
    print("Testing DMIAD with MobileNet...")
    
    # Create config for MobileNet
    config = SimpleNamespace()
    config.MODEL = SimpleNamespace()
    config.MODEL.NAME = 'dmiad'
    config.MODEL.BACKBONE = 'mobilenet_v2'  # Can be 'mobilenet_v2', 'mobilenet_v3', 'mobilenet_v2_0.5', etc.
    config.MODEL.PRETRAINED = False
    config.MODEL.PRETRAINED_PATH = None
    
    config.MODEL.MEMORY = SimpleNamespace()
    config.MODEL.MEMORY.USE_SPATIAL = True
    config.MODEL.MEMORY.FUSION_METHOD = 'add'
    config.MODEL.MEMORY.TEMPORAL_DIM = 1000  # Smaller for MobileNet
    config.MODEL.MEMORY.SPATIAL_DIM = 1000
    config.MODEL.MEMORY.SHRINK_THRES = 0.0025
    
    config.DATASET = SimpleNamespace()
    config.DATASET.CROP_SIZE = 288
    
    # Build model
    model = build_dmiad_model(config)
    print(f"✓ MobileNet DMIAD model created")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 288, 288)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✓ Input shape: {dummy_input.shape}")
        print(f"✓ Reconstructed shape: {output['reconstructed'].shape}")
        
        # Test anomaly scoring
        scores = model.compute_anomaly_score(dummy_input, output['reconstructed'])
        print(f"✓ Image scores shape: {scores['image_scores'].shape}")
        print(f"✓ Pixel scores shape: {scores['pixel_scores'].shape}")
        
        # Test memory items
        memory_items = model.get_memory_items()
        for name, item in memory_items.items():
            print(f"✓ {name} shape: {item.shape}")
    
    print("✓ MobileNet DMIAD test completed successfully!")
    
    return model


# Compare model sizes
def compare_backbone_efficiency():
    """Compare efficiency between WideResNet and MobileNet backbones"""
    print("\n=== Backbone Efficiency Comparison ===")
    
    # Test both backbones
    configs = {
        'WideResNet': {
            'backbone': 'wide_resnet',
            'feature_dim': 512,
            'memory_dim': 2000
        },
        'MobileNetV2': {
            'backbone': 'mobilenet_v2', 
            'feature_dim': 256,
            'memory_dim': 1000
        }
    }
    
    for name, cfg in configs.items():
        print(f"\n{name}:")
        
        # Create config
        config = SimpleNamespace()
        config.MODEL = SimpleNamespace()
        config.MODEL.NAME = 'dmiad'
        config.MODEL.BACKBONE = cfg['backbone']
        config.MODEL.PRETRAINED = False
        
        config.MODEL.MEMORY = SimpleNamespace()
        config.MODEL.MEMORY.USE_SPATIAL = True
        config.MODEL.MEMORY.FUSION_METHOD = 'add'
        config.MODEL.MEMORY.TEMPORAL_DIM = cfg['memory_dim']
        config.MODEL.MEMORY.SPATIAL_DIM = cfg['memory_dim']
        config.MODEL.MEMORY.SHRINK_THRES = 0.0025
        
        config.DATASET = SimpleNamespace()
        config.DATASET.CROP_SIZE = 288
        
        # Build and test model
        try:
            model = build_dmiad_model(config)
            total_params = sum(p.numel() for p in model.parameters())
            
            # Test inference time
            dummy_input = torch.randn(1, 3, 288, 288)
            
            import time
            model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = model(dummy_input)
                
                # Measure time
                start_time = time.time()
                for _ in range(100):
                    _ = model(dummy_input)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 100 * 1000  # ms
            
            print(f"  Parameters: {total_params:,}")
            print(f"  Inference time: {avg_time:.2f} ms")
            print(f"  Memory dimension: {cfg['memory_dim']}")
            
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    # Test MobileNet DMIAD
    model = test_dmiad_with_mobilenet()
    
    # Compare efficiency
    compare_backbone_efficiency()