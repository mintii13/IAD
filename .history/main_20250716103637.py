"""
DMIAD (Dual Memory Image Anomaly Detection) - Main Entry Point
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

from config.base_config import get_config
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
    parser.add_argument('--backbone', type=str, default='wide_resnet',
                       help='Backbone architecture')
    parser.add_argument('--mem_dim', type=int, default=2000,
                       help='Memory dimension')
    parser.add_argument('--use_spatial_memory', action='store_true',
                       help='Use spatial memory module')
    parser.add_argument('--fusion_method', type=str, choices=['add', 'concat', 'avg'], 
                       default='add', help='Memory fusion method')
    
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


def load_config_from_file(config_path):
    """Load configuration from file if provided"""
    # TODO: Implement YAML/JSON config loading if needed
    # For now, return None to use command line args
    return None


def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")
    
    # Load config from file if provided, otherwise use command line args
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        # TODO: Implement config file loading
        # For now, fall back to command line args
        config = get_config(args)
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
    print(f"Epochs: {config.TRAIN.EPOCHS}")
    print(f"Batch size: {config.TRAIN.BATCH_SIZE}")
    print(f"Memory dim: {config.MODEL.MEMORY.TEMPORAL_DIM}")
    print(f"Use spatial memory: {config.MODEL.MEMORY.USE_SPATIAL}")
    print(f"Fusion method: {config.MODEL.MEMORY.FUSION_METHOD}")
    print(f"Output dir: {exp_dir}")
    print("="*60)
    
    # Run training or testing
    try:
        if args.mode == 'train':
            print("Starting training...")
            train_model(config, device, exp_dir)
            print("Training completed successfully!")
        elif args.mode == 'test':
            print("Starting testing...")
            test_model(config, device, exp_dir)
            print("Testing completed successfully!")
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    except Exception as e:
        print(f"Error during {args.mode}: {str(e)}")
        raise


if __name__ == '__main__':
    main()