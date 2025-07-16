"""
MobileNet Backbone for DMIAD
Feature extractor based on MobileNetV2 with multi-scale outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import logging


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        # Batch normalization and activation
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class InvertedResidual(nn.Module):
    """Inverted Residual Block (MobileNetV2 building block)"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        layers = []
        
        # Expand phase (1x1 conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise phase (3x3 depthwise conv)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        # Project phase (1x1 conv, no activation)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2FeatureExtractor(nn.Module):
    """
    MobileNetV2 Feature Extractor for DMIAD
    Extracts multi-scale features for dual memory processing
    """
    
    def __init__(self, pretrained=True, pretrained_path=None, width_mult=1.0):
        super(MobileNetV2FeatureExtractor, self).__init__()
        
        self.width_mult = width_mult
        
        # Define channel configurations
        # [expansion_factor, channels, num_blocks, stride]
        self.cfg = [
            [1, 16, 1, 1],   # Initial block
            [6, 24, 2, 2],   # s2 features (stride 2)
            [6, 32, 3, 2],   # s4 features (stride 4) 
            [6, 64, 4, 2],   # s8 features (stride 8)
            [6, 96, 3, 1],   # Additional depth
            [6, 160, 3, 2],  # s16 features (stride 16)
            [6, 320, 1, 1],  # Final features
        ]
        
        # Build the network
        self._build_network()
        
        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights(pretrained_path)
    
    def _make_divisible(self, v, divisor=8, min_value=None):
        """Make channel number divisible by divisor"""
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    def _build_network(self):
        """Build MobileNetV2 network"""
        
        # First convolution layer
        input_channel = self._make_divisible(32 * self.width_mult)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )
        
        # Build inverted residual blocks
        features = []
        
        for t, c, n, s in self.cfg:
            output_channel = self._make_divisible(c * self.width_mult)
            
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
        
        self.features = nn.Sequential(*features)
        
        # Feature extraction points for different scales
        # s2: after block 1 (channels=24)
        # s4: after block 3 (channels=32) 
        # s8: after block 7 (channels=64)
        
        # Additional projection layers to standardize channels
        self.s2_proj = nn.Conv2d(self._make_divisible(24 * self.width_mult), 128, 1, bias=False)
        self.s4_proj = nn.Conv2d(self._make_divisible(32 * self.width_mult), 256, 1, bias=False)
        self.s8_proj = nn.Conv2d(self._make_divisible(64 * self.width_mult), 512, 1, bias=False)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _load_pretrained_weights(self, pretrained_path=None):
        """Load pretrained weights"""
        try:
            if pretrained_path and torch.os.path.exists(pretrained_path):
                # Load from custom path
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(checkpoint, strict=False)
                logging.info(f"Loaded pretrained weights from {pretrained_path}")
            else:
                # Load from torchvision
                mobilenet = models.mobilenet_v2(pretrained=True)
                
                # Extract compatible weights
                pretrained_dict = mobilenet.state_dict()
                model_dict = self.state_dict()
                
                # Filter out incompatible keys
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() 
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                
                # Update model dict
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict, strict=False)
                
                logging.info("Loaded pretrained MobileNetV2 weights from torchvision")
                
        except Exception as e:
            logging.warning(f"Failed to load pretrained weights: {e}")
            logging.info("Training from scratch")
    
    def forward(self, x):
        """
        Extract multi-scale features
        
        Args:
            x: Input tensor [N, 3, H, W]
            
        Returns:
            Dict with features at different scales:
            - s2: [N, 128, H/2, W/2]
            - s4: [N, 256, H/4, W/4]  
            - s8: [N, 512, H/8, W/8]
        """
        # Initial convolution: [N, 3, H, W] -> [N, 32, H/2, W/2]
        x = self.first_conv(x)
        
        # Progressive feature extraction
        features = {}
        
        # Extract features through the network
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Capture features at different scales
            if i == 1:  # After 2nd inverted residual block
                s2_features = self.s2_proj(x)  # [N, 128, H/2, W/2]
                features['s2'] = s2_features
                
            elif i == 3:  # After 4th inverted residual block  
                s4_features = self.s4_proj(x)  # [N, 256, H/4, W/4]
                features['s4'] = s4_features
                
            elif i == 7:  # After 8th inverted residual block
                s8_features = self.s8_proj(x)  # [N, 512, H/8, W/8]
                features['s8'] = s8_features
        
        return features


class MobileNetV3FeatureExtractor(nn.Module):
    """
    MobileNetV3 Feature Extractor (Alternative implementation)
    Uses MobileNetV3-Large architecture
    """
    
    def __init__(self, pretrained=True, pretrained_path=None):
        super(MobileNetV3FeatureExtractor, self).__init__()
        
        if pretrained and not pretrained_path:
            # Use torchvision MobileNetV3
            self.backbone = models.mobilenet_v3_large(pretrained=True)
            
            # Remove classifier
            self.backbone.classifier = nn.Identity()
            
            # Get feature extraction layers
            self.features = self.backbone.features
            
        else:
            # Custom implementation or load from path
            # For simplicity, fallback to MobileNetV2
            logging.warning("Custom MobileNetV3 not implemented, using MobileNetV2")
            self.__class__ = MobileNetV2FeatureExtractor
            self.__init__(pretrained, pretrained_path)
            return
        
        # Feature projection layers
        self.s2_proj = nn.Conv2d(40, 128, 1, bias=False)
        self.s4_proj = nn.Conv2d(80, 256, 1, bias=False)
        self.s8_proj = nn.Conv2d(160, 512, 1, bias=False)
    
    def forward(self, x):
        """Extract multi-scale features from MobileNetV3"""
        features = {}
        
        # Extract features through MobileNetV3 layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Capture features at different scales based on MobileNetV3 architecture
            if i == 3:  # Early features for s2
                s2_features = self.s2_proj(x)
                features['s2'] = s2_features
            elif i == 6:  # Mid features for s4  
                s4_features = self.s4_proj(x)
                features['s4'] = s4_features
            elif i == 12:  # Deep features for s8
                s8_features = self.s8_proj(x)
                features['s8'] = s8_features
        
        return features


def build_mobilenet_backbone(version='v2', pretrained=True, pretrained_path=None, width_mult=1.0):
    """
    Build MobileNet backbone for DMIAD
    
    Args:
        version: 'v2' or 'v3' for MobileNet version
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to custom pretrained weights
        width_mult: Width multiplier for MobileNetV2
    
    Returns:
        MobileNet feature extractor
    """
    if version == 'v2':
        return MobileNetV2FeatureExtractor(
            pretrained=pretrained, 
            pretrained_path=pretrained_path,
            width_mult=width_mult
        )
    elif version == 'v3':
        return MobileNetV3FeatureExtractor(
            pretrained=pretrained,
            pretrained_path=pretrained_path
        )
    else:
        raise ValueError(f"Unsupported MobileNet version: {version}")


# Lightweight MobileNet decoder for faster inference
class MobileNetDecoder(nn.Module):
    """
    Lightweight decoder using depthwise separable convolutions
    Compatible with MobileNet encoder
    """
    
    def __init__(self, input_dim=512):
        super(MobileNetDecoder, self).__init__()
        
        self.input_dim = input_dim
        
        # Upsampling path with depthwise separable convolutions
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 256, 2, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True)
        )
        
        self.conv1 = DepthwiseSeparableConv(256 + 256, 256)  # Skip connection from s4
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True)
        )
        
        self.conv2 = DepthwiseSeparableConv(128 + 128, 128)  # Skip connection from s2
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Sequential(
            DepthwiseSeparableConv(64, 32),
            nn.Conv2d(32, 3, 3, padding=1, bias=False)
        )
        
        # Channel attention modules
        self.att_s8 = ChannelAttention(input_dim)
        self.att_s4 = ChannelAttention(256)
        self.att_s2 = ChannelAttention(128)
    
    def forward(self, memory_features, skip_features):
        """
        Decode features with skip connections
        
        Args:
            memory_features: Features from memory module [B, input_dim, H/8, W/8]
            skip_features: Dict with skip connections
                - s2: [B, 128, H/2, W/2]
                - s4: [B, 256, H/4, W/4]
        
        Returns:
            Reconstructed image [B, 3, H, W]
        """
        x = memory_features
        
        # Apply attention to memory features
        x = self.att_s8(x)
        
        # Upsample to s4 resolution
        x = self.up1(x)  # [B, 256, H/4, W/4]
        
        # Skip connection from s4
        x = torch.cat([x, skip_features['s4']], dim=1)  # [B, 512, H/4, W/4]
        x = self.conv1(x)  # [B, 256, H/4, W/4]
        x = self.att_s4(x)
        
        # Upsample to s2 resolution  
        x = self.up2(x)  # [B, 128, H/2, W/2]
        
        # Skip connection from s2
        x = torch.cat([x, skip_features['s2']], dim=1)  # [B, 256, H/2, W/2]
        x = self.conv2(x)  # [B, 128, H/2, W/2]
        x = self.att_s2(x)
        
        # Upsample to original resolution
        x = self.up3(x)  # [B, 64, H, W]
        
        # Final reconstruction
        x = self.final(x)  # [B, 3, H, W]
        
        return x


class ChannelAttention(nn.Module):
    """Channel Attention Module for MobileNet decoder"""
    def __init__(self, input_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        reduced_channels = max(input_channels // reduction, 8)
        
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, reduced_channels, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(reduced_channels, input_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.layer(y)
        return x * y


# Test function
def test_mobilenet_backbone():
    """Test MobileNet backbone functionality"""
    print("Testing MobileNet Backbone...")
    
    # Test MobileNetV2
    backbone_v2 = build_mobilenet_backbone('v2', pretrained=False)
    
    # Test input
    x = torch.randn(2, 3, 288, 288)
    
    with torch.no_grad():
        features = backbone_v2(x)
        
        print("MobileNetV2 Feature Extraction:")
        for scale, feat in features.items():
            print(f"  {scale}: {feat.shape}")
    
    # Test decoder
    decoder = MobileNetDecoder(input_dim=512)
    
    with torch.no_grad():
        # Simulate memory-processed features
        memory_features = torch.randn(2, 512, 36, 36)  # H/8, W/8
        
        reconstructed = decoder(memory_features, features)
        print(f"Reconstructed: {reconstructed.shape}")
    
    print("✓ MobileNet backbone test completed!")


if __name__ == "__main__":
    test_mobilenet_backbone()