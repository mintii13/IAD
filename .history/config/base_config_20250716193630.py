"""
Updated Base Configuration for DMIAD with MobileNet Support
"""

import os
from easydict import EasyDict as edict


def get_config(args):
    """Get configuration based on arguments"""
    config = edict()
    
    # Dataset configuration
    config.DATASET = edict()
    config.DATASET.NAME = args.dataset
    config.DATASET.ROOT = args.data_path
    config.DATASET.CLASS_NAME = getattr(args, 'class_name', None)
    config.DATASET.SETTING = args.setting  # 'single' or 'multi'
    config.DATASET.IMAGE_SIZE = (256, 256)
    config.DATASET.RESIZE = 329
    config.DATASET.CROP_SIZE = 288
    
    # Model configuration
    config.MODEL = edict()
    config.MODEL.NAME = 'dmiad'
    config.MODEL.BACKBONE = args.backbone
    config.MODEL.PRETRAINED = True
    
    # Backbone-specific configurations
    if args.backbone.startswith('mobilenet'):
        # MobileNet specific settings
        config.MODEL.MOBILENET = edict()
        
        # Parse MobileNet version
        if 'v2' in args.backbone:
            config.MODEL.MOBILENET.VERSION = 'v2'
        elif 'v3' in args.backbone:
            config.MODEL.MOBILENET.VERSION = 'v3'
        else:
            config.MODEL.MOBILENET.VERSION = 'v2'  # default
        
        # Parse width multiplier from backbone name
        config.MODEL.MOBILENET.WIDTH_MULT = 1.0
        if '_' in args.backbone:
            parts = args.backbone.split('_')
            for part in parts:
                if part.replace('.', '').isdigit():
                    config.MODEL.MOBILENET.WIDTH_MULT = float(part)
                    break
    
    # Memory configuration
    config.MODEL.MEMORY = edict()
    config.MODEL.MEMORY.TEMPORAL_DIM = args.mem_dim
    config.MODEL.MEMORY.SPATIAL_DIM = args.mem_dim
    config.MODEL.MEMORY.USE_SPATIAL = args.use_spatial_memory
    config.MODEL.MEMORY.FUSION_METHOD = args.fusion_method
    config.MODEL.MEMORY.SHRINK_THRES = 0.0025
    config.MODEL.MEMORY.NORMALIZE_MEMORY = False
    config.MODEL.MEMORY.NORMALIZE_QUERY = False
    config.MODEL.MEMORY.USE_SHARED_MLP = False
    
    # Training configuration
    config.TRAIN = edict()
    config.TRAIN.BATCH_SIZE = args.batch_size
    config.TRAIN.EPOCHS = args.epochs
    config.TRAIN.LR = args.lr
    config.TRAIN.WEIGHT_DECAY = 1e-4
    config.TRAIN.SCHEDULER = 'cosine'
    config.TRAIN.WARMUP_EPOCHS = 10
    
    # Backbone-specific training adjustments
    if args.backbone.startswith('mobilenet'):
        # Adjust learning rate for MobileNet (typically needs higher LR)
        config.TRAIN.LR = args.lr * 2.0
        # Reduce weight decay for smaller model
        config.TRAIN.WEIGHT_DECAY = 5e-5
    
    # Loss configuration
    config.LOSS = edict()
    config.LOSS.RECONSTRUCTION_WEIGHT = 1.0
    config.LOSS.MEMORY_WEIGHT = 0.01
    config.LOSS.SPARSITY_WEIGHT = 0.0001
    
    # Test configuration
    config.TEST = edict()
    config.TEST.BATCH_SIZE = 1
    config.TEST.SAVE_IMAGES = getattr(args, 'save_images', False)
    config.TEST.CHECKPOINT = getattr(args, 'checkpoint', None)
    
    # Augmentation configuration
    config.AUG = edict()
    config.AUG.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    config.AUG.NORMALIZE_STD = [0.229, 0.224, 0.225]
    config.AUG.HORIZONTAL_FLIP = 0.5
    config.AUG.ROTATION = 10
    config.AUG.COLOR_JITTER = 0.1
    
    # Output configuration
    config.OUTPUT_DIR = args.output_dir
    config.EXP_NAME = args.exp_name
    config.SAVE_FREQ = 10  # Save checkpoint every N epochs
    config.EVAL_FREQ = 5   # Evaluate every N epochs
    
    # Device configuration
    config.DEVICE = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    config.NUM_WORKERS = 4
    config.PIN_MEMORY = True
    
    return config


def get_mobilenet_config(version='v2', width_mult=1.0, memory_dim=1000):
    """Get MobileNet specific configuration"""
    config = edict()
    
    config.VERSION = version
    config.WIDTH_MULT = width_mult
    config.PRETRAINED = True
    
    # Recommended memory dimensions based on model size
    if width_mult <= 0.5:
        config.RECOMMENDED_MEMORY_DIM = max(500, memory_dim // 2)
    elif width_mult <= 0.75:
        config.RECOMMENDED_MEMORY_DIM = max(750, memory_dim // 1.5)
    else:
        config.RECOMMENDED_MEMORY_DIM = memory_dim
    
    # Recommended training settings
    config.RECOMMENDED_BATCH_SIZE = 16 if width_mult <= 0.5 else 8
    config.RECOMMENDED_LR = 2e-4 if width_mult <= 0.5 else 1e-4
    
    return config


def get_backbone_specific_config(backbone_name):
    """Get backbone-specific optimized configuration"""
    
    if backbone_name.startswith('mobilenet'):
        # MobileNet configurations
        if backbone_name == 'mobilenet_v2':
            return {
                'memory_dim': 1000,
                'feature_dim': 256,
                'batch_size': 8,
                'lr_multiplier': 2.0,
                'weight_decay': 5e-5
            }
        elif backbone_name == 'mobilenet_v2_0.5':
            return {
                'memory_dim': 500,
                'feature_dim': 128,
                'batch_size': 16,
                'lr_multiplier': 3.0,
                'weight_decay': 1e-5
            }
        elif backbone_name == 'mobilenet_v3':
            return {
                'memory_dim': 1200,
                'feature_dim': 256,
                'batch_size': 8,
                'lr_multiplier': 1.5,
                'weight_decay': 5e-5
            }
    
    elif backbone_name == 'wide_resnet':
        # Original WideResNet configuration
        return {
            'memory_dim': 2000,
            'feature_dim': 512,
            'batch_size': 6,
            'lr_multiplier': 1.0,
            'weight_decay': 1e-4
        }
    
    # Default configuration
    return {
        'memory_dim': 1000,
        'feature_dim': 256,
        'batch_size': 8,
        'lr_multiplier': 1.0,
        'weight_decay': 1e-4
    }


def auto_adjust_config_for_backbone(config):
    """Automatically adjust configuration based on backbone"""
    
    backbone_cfg = get_backbone_specific_config(config.MODEL.BACKBONE)
    
    # Adjust memory dimensions if not explicitly set
    if hasattr(config.MODEL.MEMORY, 'AUTO_ADJUST') and config.MODEL.MEMORY.AUTO_ADJUST:
        config.MODEL.MEMORY.TEMPORAL_DIM = backbone_cfg['memory_dim']
        config.MODEL.MEMORY.SPATIAL_DIM = backbone_cfg['memory_dim']
    
    # Adjust training parameters
    config.TRAIN.LR *= backbone_cfg['lr_multiplier']
    config.TRAIN.WEIGHT_DECAY = backbone_cfg['weight_decay']
    
    # Adjust batch size if not specified
    if not hasattr(config.TRAIN, 'BATCH_SIZE_OVERRIDE'):
        config.TRAIN.BATCH_SIZE = backbone_cfg['batch_size']
    
    return config


# Dataset specific configurations (unchanged)
def get_mvtec_config():
    """MVTec specific configuration"""
    config = edict()
    config.CLASSES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    return config


def get_visa_config():
    """VisA specific configuration"""
    config = edict()
    config.CLASSES = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
    ]
    return config


def get_mpdd_config():
    """MPDD specific configuration"""
    config = edict()
    config.CLASSES = [
        'bracket_black', 'bracket_brown', 'bracket_white',
        'connector', 'metal_plate', 'tubes'
    ]
    return config


def get_itdd_config():
    """ITDD specific configuration"""
    config = edict()
    config.CLASSES = [
        'cotton_fabric', 'dyed_fabric', 'hemp_fabric', 'plaid_fabric'
    ]
    return config


# Utility function to create optimized config for specific backbone
def create_optimized_config_for_backbone(args, backbone_name):
    """Create optimized configuration for specific backbone"""
    
    # Get base config
    args.backbone = backbone_name
    config = get_config(args)
    
    # Get backbone-specific optimizations
    backbone_cfg = get_backbone_specific_config(backbone_name)
    
    # Apply optimizations
    if not hasattr(args, 'mem_dim_override'):
        config.MODEL.MEMORY.TEMPORAL_DIM = backbone_cfg['memory_dim']
        config.MODEL.MEMORY.SPATIAL_DIM = backbone_cfg['memory_dim']
    
    config.TRAIN.LR = args.lr * backbone_cfg['lr_multiplier']
    config.TRAIN.WEIGHT_DECAY = backbone_cfg['weight_decay']
    
    if not hasattr(args, 'batch_size_override'):
        config.TRAIN.BATCH_SIZE = backbone_cfg['batch_size']
    
    return config


# Example configurations for different use cases
def get_fast_inference_config(args):
    """Configuration optimized for fast inference"""
    args.backbone = 'mobilenet_v2_0.5'  # Smallest, fastest model
    config = create_optimized_config_for_backbone(args, args.backbone)
    
    # Further optimizations for speed
    config.MODEL.MEMORY.TEMPORAL_DIM = 500
    config.MODEL.MEMORY.SPATIAL_DIM = 500
    config.TEST.BATCH_SIZE = 4  # Larger batch for faster processing
    
    return config


def get_high_accuracy_config(args):
    """Configuration optimized for high accuracy"""
    args.backbone = 'wide_resnet'  # Most accurate backbone
    config = create_optimized_config_for_backbone(args, args.backbone)
    
    # Optimizations for accuracy
    config.MODEL.MEMORY.TEMPORAL_DIM = 3000  # Larger memory
    config.MODEL.MEMORY.SPATIAL_DIM = 3000
    config.TRAIN.EPOCHS = 200  # More training
    config.TRAIN.LR = args.lr * 0.5  # Lower learning rate for stability
    
    return config


def get_balanced_config(args):
    """Configuration balanced between speed and accuracy"""
    args.backbone = 'mobilenet_v2'  # Good balance
    config = create_optimized_config_for_backbone(args, args.backbone)
    
    # Balanced settings
    config.MODEL.MEMORY.TEMPORAL_DIM = 1000
    config.MODEL.MEMORY.SPATIAL_DIM = 1000
    
    return config