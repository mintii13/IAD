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
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Mode: train or test')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device id')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['mvtec', 'visa', 'mpdd', 'itdd'],
                       help='Dataset name')
    parser.add_argument('--data_path', type=str,
                       help='Path to dataset')
    parser.add_argument('--class_name', type=str,
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


def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = get_config(args)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Run training or testing
    if args.mode == 'train':
        print("Starting training...")
        train_model(config, device, exp_dir)
    elif args.mode == 'test':
        print("Starting testing...")
        test_model(config, device, exp_dir)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()