"""
Base Configuration for DMIAD
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


def merge_config_file(config, cfg_file):
    """Merge config from file"""
    # TODO: Implement config file loading if needed
    pass


# Dataset specific configurations
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