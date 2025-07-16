"""
DMIAD (Dual Memory Image Anomaly Detection) - Main Entry Point
Updated with MobileNet Support
"""

import argparse
import os
import sys
import torch
import random
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.base_config import get_config, create_optimized_config_for_backbone
from train import train_model
from test import test_model


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DMIAD - Dual Memory Image Anomaly Detection')
    
    # Basic arguments
    parser.add_argument('--config', type=str, required=False, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Mode: train or test')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device id')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['mvtec', 'visa', 'mpdd', 'itdd'],
                       required=True, help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--class_name', type=str, required=True,
                       help='Class name for single-class training')
    parser.add_argument('--setting', type=str, choices=['single', 'multi'], default='single',
                       help='Single-class or multi-class setting')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, 
                       choices=['wide_resnet', 'mobilenet_v2', 'mobilenet_v2_0.5', 'mobilenet_v2_0.75', 
                               'mobilenet_v2_1.4', 'mobilenet_v2_0.35', 'mobilenet_v3'],
                       default='wide_resnet',
                       help='Backbone architecture')
    parser.add_argument('--mem_dim', type=int, default=2000,
                       help='Memory dimension')
    parser.add_argument('--use_spatial_memory', action='store_true',
                       help='Use spatial memory module')
    parser.add_argument('--fusion_method', type=str, choices=['add', 'concat', 'weighted'], 
                       default='add', help='Memory fusion method')
    
    # Auto-optimization arguments
    parser.add_argument('--auto_optimize', action='store_true',
                       help='Automatically optimize config for selected backbone')
    parser.add_argument('--profile', choices=['fast', 'balanced', 'accurate'],
                       help='Use predefined optimization profile')
    
    # Test arguments
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint for testing')
    parser.add_argument('--save_images', action='store_true',
                       help='Save visualization images during testing')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--exp_name', type=str, default='dmiad_exp',
                       help='Experiment name')
    
    return parser.parse_args()


def apply_optimization_profile(args):
    """Apply predefined optimization profiles"""
    if not args.profile:
        return args
    
    if args.profile == 'fast':
        # Fast inference profile
        args.backbone = 'mobilenet_v2_0.5'
        args.mem_dim = 500
        args.batch_size = 16
        args.epochs = 200
        args.lr = 3e-4
        
    elif args.profile == 'balanced':
        # Balanced profile
        args.backbone = 'mobilenet_v2'
        args.mem_dim = 1000
        args.batch_size = 8
        args.epochs = 300
        args.lr = 2e-4
        
    elif args.profile == 'accurate':
        # High accuracy profile
        args.backbone = 'wide_resnet'
        args.mem_dim = 3000
        args.batch_size = 6
        args.epochs = 400
        args.lr = 8e-5
    
    print(f"Applied {args.profile} optimization profile:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Memory dim: {args.mem_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    
    return args


def get_model_complexity_info(config):
    """Estimate model complexity"""
    
    # Rough parameter estimation
    if config.MODEL.BACKBONE.startswith('mobilenet'):
        if 'v2_0.35' in config.MODEL.BACKBONE:
            backbone_params = 1.7e6  # ~1.7M parameters
        elif 'v2_0.5' in config.MODEL.BACKBONE:
            backbone_params = 1.95e6  # ~1.95M parameters
        elif 'v2_0.75' in config.MODEL.BACKBONE:
            backbone_params = 2.6e6   # ~2.6M parameters
        elif 'v2_1.4' in config.MODEL.BACKBONE:
            backbone_params = 6.9e6   # ~6.9M parameters
        elif 'v3' in config.MODEL.BACKBONE:
            backbone_params = 5.4e6   # ~5.4M parameters
        else:  # mobilenet_v2
            backbone_params = 3.4e6   # ~3.4M parameters
            
        feature_dim = 256
    else:  # wide_resnet
        backbone_params = 126.8e6  # ~126.8M parameters
        feature_dim = 512
    
    # Memory parameters
    temporal_memory_params = config.MODEL.MEMORY.TEMPORAL_DIM * feature_dim
    spatial_memory_params = config.MODEL.MEMORY.SPATIAL_DIM * config.DATASET.CROP_SIZE // 8 * config.DATASET.CROP_SIZE // 8
    
    # Decoder parameters (rough estimate)
    decoder_params = feature_dim * 256 + 256 * 128 + 128 * 64 + 64 * 32 + 32 * 3
    
    total_params = backbone_params + temporal_memory_params + spatial_memory_params + decoder_params
    
    return {
        'backbone_params': backbone_params,
        'temporal_memory_params': temporal_memory_params,
        'spatial_memory_params': spatial_memory_params,
        'decoder_params': decoder_params,
        'total_params': total_params,
        'total_params_m': total_params / 1e6
    }


def print_model_info(config):
    """Print model information"""
    complexity = get_model_complexity_info(config)
    
    print("\n" + "="*60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("="*60)
    print(f"Backbone: {config.MODEL.BACKBONE}")
    print(f"  Parameters: {complexity['backbone_params']/1e6:.1f}M")
    print(f"Memory Modules:")
    print(f"  Temporal: {complexity['temporal_memory_params']/1e6:.1f}M")
    print(f"  Spatial: {complexity['spatial_memory_params']/1e6:.1f}M")
    print(f"Decoder: {complexity['decoder_params']/1e6:.1f}M")
    print(f"Total Estimated Parameters: {complexity['total_params_m']:.1f}M")
    
    # Memory requirements estimation
    if config.MODEL.BACKBONE.startswith('mobilenet'):
        estimated_gpu_memory = complexity['total_params_m'] * 4 * 2  # FP32 * 2 for gradients
        print(f"Estimated GPU Memory: ~{estimated_gpu_memory:.0f}MB")
    else:
        estimated_gpu_memory = complexity['total_params_m'] * 4 * 2
        print(f"Estimated GPU Memory: ~{estimated_gpu_memory:.0f}MB")
    
    print("="*60)


def main():
    """Main function"""
    args = parse_args()
    
    # Apply optimization profile if specified
    args = apply_optimization_profile(args)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        # TODO: Implement config file loading
        config = get_config(args)
    elif args.auto_optimize:
        # Auto-optimize config for backbone
        print(f"Auto-optimizing config for {args.backbone}")
        config = create_optimized_config_for_backbone(args, args.backbone)
    else:
        # Use command line arguments to build config
        config = get_config(args)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Print configuration summary
    print("\n" + "="*60)
    print("DMIAD Configuration Summary")
    print("="*60)
    print(f"Dataset: {config.DATASET.NAME}")
    print(f"Class: {config.DATASET.CLASS_NAME}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Backbone: {config.MODEL.BACKBONE}")
    print(f"Epochs: {config.TRAIN.EPOCHS}")
    print(f"Batch size: {config.TRAIN.BATCH_SIZE}")
    print(f"Learning rate: {config.TRAIN.LR}")
    print(f"Memory dim: {config.MODEL.MEMORY.TEMPORAL_DIM}")
    print(f"Use spatial memory: {config.MODEL.MEMORY.USE_SPATIAL}")
    print(f"Fusion method: {config.MODEL.MEMORY.FUSION_METHOD}")
    print(f"Output dir: {exp_dir}")
    
    # Print model complexity info
    print_model_info(config)
    
    # Validate GPU memory if using CUDA
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(device.index).total_memory / 1e9
        print(f"Available GPU Memory: {gpu_memory:.1f}GB")
        
        complexity = get_model_complexity_info(config)
        if complexity['total_params_m'] > 100 and gpu_memory < 8:
            print("⚠️  Warning: Large model may not fit in GPU memory. Consider using MobileNet backbone.")
    
    # Run training or testing
    try:
        if args.mode == 'train':
            print("\nStarting training...")
            train_model(config, device, exp_dir)
            print("Training completed successfully!")
        elif args.mode == 'test':
            print("\nStarting testing...")
            test_model(config, device, exp_dir)
            print("Testing completed successfully!")
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    except Exception as e:
        print(f"Error during {args.mode}: {str(e)}")
        raise


if __name__ == '__main__':
    main()