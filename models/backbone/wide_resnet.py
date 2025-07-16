"""
WideResNet Backbone for DMIAD
Extracted from the provided code
"""

import logging
import sys
from collections import OrderedDict
from functools import partial
import torch.nn as nn
import torch

Norm2d = nn.BatchNorm2d

def bnrelu(channels):
    """Single Layer BN and ReLU"""
    return nn.Sequential(Norm2d(channels), nn.ReLU(inplace=True))

class GlobalAvgPool2d(nn.Module):
    """Global average pooling over the input's spatial dimensions"""
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        logging.info("Global Average Pooling Initialized")

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)

class IdentityResidualBlock(nn.Module):
    """Identity Residual Block for WideResnet"""
    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=1,
                 norm_act=bnrelu, dropout=None, dist_bn=False):
        super(IdentityResidualBlock, self).__init__()
        self.dist_bn = dist_bn

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=stride,
                                    padding=dilation, bias=False, dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1,
                                    padding=dilation, bias=False, dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=stride,
                                    padding=0, bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1,
                                    padding=dilation, bias=False, groups=groups, dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)
        return out

class WiderResNetA2(nn.Module):
    """Wider ResNet with pre-activation (identity mapping) blocks"""
    def __init__(self, structure, norm_act=bnrelu, classes=0, dilation=False, dist_bn=False):
        super(WiderResNetA2, self).__init__()
        self.dist_bn = dist_bn
        self.structure = structure
        self.dilation = dilation

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = torch.nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        ]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), 
                   (512, 1024, 2048), (1024, 2048, 4096)]
        
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1

                if mod_id == 4:
                    drop = partial(nn.Dropout, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout, p=0.5)
                else:
                    drop = None

                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels, channels[mod_id], 
                                        norm_act=norm_act, stride=stride, 
                                        dilation=dil, dropout=drop, dist_bn=self.dist_bn)
                ))
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id < 2:
                self.add_module("pool%d" % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))   # s2
        out = self.mod3(self.pool3(out))   # s4
        out = self.mod4(out)               # s8
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, "classifier"):
            return self.classifier(out)
        return out

# Network definitions
_NETS = {
    "16": {"structure": [1, 1, 1, 1, 1, 1]},
    "20": {"structure": [1, 1, 1, 3, 1, 1]},
    "38": {"structure": [3, 3, 6, 3, 1, 1]},
}

__all__ = []
for name, params in _NETS.items():
    net_name = "wider_resnet" + name + "_a2"
    setattr(sys.modules[__name__], net_name, partial(WiderResNetA2, **params))
    __all__.append(net_name)

class WideResNetFeatureExtractor(nn.Module):
    """
    WideResNet Feature Extractor for DMIAD
    Extracts multi-scale features for dual memory processing
    """
    def __init__(self, pretrained=False, pretrained_path=None):
        super(WideResNetFeatureExtractor, self).__init__()
        
        # Create WideResNet38
        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        
        # Load pretrained weights if specified
        if pretrained and pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            wide_resnet.load_state_dict(checkpoint['state_dict'])
            del checkpoint
        
        # Extract modules for feature extraction
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        
        del wide_resnet

    def forward(self, x):
        """
        Extract multi-scale features
        Returns features at different scales for skip connections and memory processing
        """
        x = self.mod1(x)                    # [N, 64, H, W]
        
        x = self.mod2(self.pool2(x))        # [N, 128, H/2, W/2]
        s2_features = x
        
        x = self.mod3(self.pool3(x))        # [N, 256, H/4, W/4]  
        s4_features = x
        
        x = self.mod4(x)                    # [N, 1024, H/8, W/8]
        x = self.mod5(x)                    # [N, 2048, H/8, W/8]
        x = self.mod6(x)                    # [N, 4096, H/8, W/8]
        x = self.mod7(x)                    # [N, 4096, H/8, W/8]
        s8_features = x
        
        return {
            's2': s2_features,      # [N, 128, H/2, W/2]
            's4': s4_features,      # [N, 256, H/4, W/4] 
            's8': s8_features,      # [N, 4096, H/8, W/8] - for memory processing
        }

# Utility function to create backbone
def build_wideresnet_backbone(pretrained=False, pretrained_path=None):
    """Build WideResNet backbone for DMIAD"""
    return WideResNetFeatureExtractor(pretrained=pretrained, pretrained_path=pretrained_path)