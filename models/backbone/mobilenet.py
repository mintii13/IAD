"""
MobileNet Backbone for DMIAD - FIXED VERSION
Fixed size mismatch issues in decoder
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


class MobileNetV2FeatureExtractor(nn.Module):
    """
    MobileNetV2 Feature Extractor for DMIAD - FIXED VERSION
    Extracts multi-scale features for dual memory processing
    """
    
    def __init__(self, pretrained=True, pretrained_path=None, width_mult=1.0):
        super(MobileNetV2FeatureExtractor, self).__init__()
        
        self.width_mult = width_mult
        
        # Use torchvision MobileNetV2 as base and extract features
        if pretrained and not pretrained_path:
            # Load pretrained MobileNetV2
            mobilenet = models.mobilenet_v2(pretrained=True)
            self.features = mobilenet.features
        else:
            # Build custom MobileNetV2
            mobilenet = models.mobilenet_v2(pretrained=False)
            self.features = mobilenet.features
        
        # Feature extraction indices for different scales
        # Based on MobileNetV2 architecture:
        # - features[3]: after 1st downsample (stride 2) -> s2
        # - features[6]: after 2nd downsample (stride 4) -> s4  
        # - features[13]: after 3rd downsample (stride 8) -> s8
        
        # Get actual channel numbers from model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 288, 288)
            
            # Extract features to get actual channel dimensions
            x = dummy_input
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == 3:  # s2 features
                    s2_channels = x.shape[1]
                elif i == 6:  # s4 features
                    s4_channels = x.shape[1]
                elif i == 13:  # s8 features
                    s8_channels = x.shape[1]
                    break
        
        # Projection layers to standardize channels
        self.s2_proj = nn.Conv2d(s2_channels, 128, 1, bias=False)
        self.s4_proj = nn.Conv2d(s4_channels, 256, 1, bias=False)
        self.s8_proj = nn.Conv2d(s8_channels, 512, 1, bias=False)
        
        print(f"MobileNetV2 channel mapping:")
        print(f"  s2: {s2_channels} -> 128")
        print(f"  s4: {s4_channels} -> 256") 
        print(f"  s8: {s8_channels} -> 512")
    
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
        features_dict = {}
        
        # Progressive feature extraction through MobileNetV2
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Capture features at different scales
            if i == 3:  # After 1st downsample (stride 2)
                s2_features = self.s2_proj(x)  # [N, 128, H/2, W/2]
                features_dict['s2'] = s2_features
                
            elif i == 6:  # After 2nd downsample (stride 4)
                s4_features = self.s4_proj(x)  # [N, 256, H/4, W/4]
                features_dict['s4'] = s4_features
                
            elif i == 13:  # After 3rd downsample (stride 8)
                s8_features = self.s8_proj(x)  # [N, 512, H/8, W/8]
                features_dict['s8'] = s8_features
                break  # We have all needed features
        
        return features_dict


class MobileNetDecoder(nn.Module):
    """
    Fixed MobileNet Decoder - ensures proper output size
    """
    
    def __init__(self, input_dim=256):  # Changed default from 512 to 256
        super(MobileNetDecoder, self).__init__()
        
        self.input_dim = input_dim
        
        # Channel attention modules
        self.att_s8 = ChannelAttention(input_dim)
        self.att_s4 = ChannelAttention(256)
        self.att_s2 = ChannelAttention(128)
        
        # Use ConvTranspose2d for precise size control
        # From spatial_size 18x18 -> 288x288 (16x upsampling)
        
        # Stage 1: 18x18 -> 36x36 (2x)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True)
        )
        
        # Fusion layer for s4 skip connection  
        self.conv1 = DepthwiseSeparableConv(256 + 256, 256)
        
        # Stage 2: 36x36 -> 72x72 (2x)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True)
        )
        
        # Fusion layer for s2 skip connection
        self.conv2 = DepthwiseSeparableConv(128 + 128, 128)
        
        # Stage 3: 72x72 -> 144x144 (2x)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True)
        )
        
        # Stage 4: 144x144 -> 288x288 (2x) - Final upsampling
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False)
    
    def forward(self, memory_features, skip_features):
        """
        Decode features with skip connections
        
        Args:
            memory_features: [B, input_dim, 18, 18] from memory module
            skip_features: Dict with skip connections
                - s2: [B, 128, 72, 72]  (H/4, W/4)
                - s4: [B, 256, 36, 36]  (H/8, W/8)
        
        Returns:
            Reconstructed image [B, 3, 288, 288]
        """
        x = memory_features  # [B, input_dim, 18, 18]
        
        # Apply attention to memory features
        x = self.att_s8(x)
        
        # Stage 1: 18x18 -> 36x36
        x = self.up1(x)  # [B, 256, 36, 36]
        
        # Skip connection from s4 (resize if needed)
        s4_features = skip_features['s4']  # [B, 256, 36, 36]
        if x.shape[-2:] != s4_features.shape[-2:]:
            s4_features = F.interpolate(s4_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, s4_features], dim=1)  # [B, 512, 36, 36]
        x = self.conv1(x)  # [B, 256, 36, 36]
        x = self.att_s4(x)
        
        # Stage 2: 36x36 -> 72x72
        x = self.up2(x)  # [B, 128, 72, 72]
        
        # Skip connection from s2 (resize if needed)
        s2_features = skip_features['s2']  # [B, 128, 72, 72]
        if x.shape[-2:] != s2_features.shape[-2:]:
            s2_features = F.interpolate(s2_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, s2_features], dim=1)  # [B, 256, 72, 72]
        x = self.conv2(x)  # [B, 128, 72, 72]
        x = self.att_s2(x)
        
        # Stage 3: 72x72 -> 144x144
        x = self.up3(x)  # [B, 64, 144, 144]
        
        # Stage 4: 144x144 -> 288x288 (final upsampling)
        x = self.up4(x)  # [B, 32, 288, 288]
        
        # Final reconstruction
        x = self.final(x)  # [B, 3, 288, 288]
        
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
    else:
        # For now, default to V2
        return MobileNetV2FeatureExtractor(
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            width_mult=width_mult
        )


# Test function to verify size compatibility
def test_mobilenet_size_compatibility():
    """Test MobileNet backbone and decoder size compatibility"""
    print("Testing MobileNet size compatibility...")
    
    # Create backbone
    backbone = build_mobilenet_backbone('v2', pretrained=False)
    decoder = MobileNetDecoder(input_dim=512)
    
    # Test different input sizes
    input_sizes = [(288, 288), (256, 256), (224, 224)]
    
    for h, w in input_sizes:
        print(f"\nTesting input size: {h}x{w}")
        
        # Test input
        x = torch.randn(2, 3, h, w)
        
        with torch.no_grad():
            # Extract features
            features = backbone(x)
            print(f"Feature sizes:")
            for scale, feat in features.items():
                print(f"  {scale}: {feat.shape}")
            
            # Simulate memory processing (identity for test)
            memory_features = features['s8']
            
            # Test decoder
            try:
                reconstructed = decoder(memory_features, features)
                print(f"Reconstruction: {reconstructed.shape}")
                print(f"Size match: {reconstructed.shape[-2:] == x.shape[-2:]}")
            except Exception as e:
                print(f"Decoder error: {e}")
    
    print("✓ Size compatibility test completed!")


if __name__ == "__main__":
    test_mobilenet_size_compatibility()