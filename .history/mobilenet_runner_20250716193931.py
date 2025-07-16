#!/usr/bin/env python3
"""
MobileNet DMIAD Runner Script
Simplified script for running DMIAD with MobileNet backbones
"""

import subprocess
import sys
import os
from datetime import datetime
import argparse

# Configuration for MobileNet DMIAD variants
MOBILENET_CONFIGS = {
    # Fast inference (smallest model)
    'fast': {
        'backbone': 'mobilenet_v2_0.5',
        'mem_dim': 500,
        'batch_size': 16,
        'epochs': 200,
        'lr': 3e-4,
        'description': 'Fast inference - MobileNetV2 0.5x width'
    },
    
    # Balanced (good speed/accuracy tradeoff) 
    'balanced': {
        'backbone': 'mobilenet_v2',
        'mem_dim': 1000,
        'batch_size': 8,
        'epochs': 300,
        'lr': 2e-4,
        'description': 'Balanced - MobileNetV2 1.0x width'
    },
    
    # High accuracy (larger MobileNet)
    'accurate': {
        'backbone': 'mobilenet_v2_1.4',
        'mem_dim': 1500,
        'batch_size': 6,
        'epochs': 400,
        'lr': 1.5e-4,
        'description': 'High accuracy - MobileNetV2 1.4x width'
    },
    
    # MobileNetV3 variant
    'v3': {
        'backbone': 'mobilenet_v3',
        'mem_dim': 1200,
        'batch_size': 8,
        'epochs': 350,
        'lr': 1.8e-4,
        'description': 'MobileNetV3-Large'
    },
    
    # Ultra light (for edge devices)
    'ultra_light': {
        'backbone': 'mobilenet_v2_0.35',
        'mem_dim': 300,
        'batch_size': 32,
        'epochs': 150,
        'lr': 5e-4,
        'description': 'Ultra light - MobileNetV2 0.35x width'
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    'mvtec': {
        'classes': [
            'carpet', 'grid', 'leather', 'tile', 'wood',  # Texture
            'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',  # Object
            'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
        ],
        'default_path': r'./dataset/MVTec'
    },
    'visa': {
        'classes': [
            'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
            'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
        ],
        'default_path': r'./dataset/VisA'
    },
    'mpdd': {
        'classes': [
            'bracket_black', 'bracket_brown', 'bracket_white',
            'connector', 'metal_plate', 'tubes'
        ],
        'default_path': r'./dataset/MPDD'
    }
}


def get_timestamp():
    """Get current timestamp for experiment naming"""
    return datetime.now().strftime("%d%m%Y_%H%M")


def run_mobilenet_dmiad(dataset, classes, config_name, data_path, mode='train', 
                       gpu=0, output_dir=None, custom_config=None):
    """
    Run DMIAD with MobileNet backbone
    
    Args:
        dataset: Dataset name ('mvtec', 'visa', 'mpdd')
        classes: List of class names to process
        config_name: MobileNet config name ('fast', 'balanced', 'accurate', etc.)
        data_path: Path to dataset
        mode: 'train' or 'test'
        gpu: GPU device ID
        output_dir: Output directory (optional)
        custom_config: Custom config override (optional)
    
    Returns:
        bool: Success status
    """
    
    # Get MobileNet configuration
    if custom_config:
        config = custom_config
    else:
        config = MOBILENET_CONFIGS[config_name].copy()
    
    # Default output directory
    if output_dir is None:
        timestamp = get_timestamp()
        output_dir = f'./results/mobilenet_{config_name}_{dataset}_{timestamp}'
    
    print(f"MobileNet DMIAD - {config['description']}")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Classes: {classes}")
    print(f"Mode: {mode.upper()}")
    print(f"Backbone: {config['backbone']}")
    print(f"Memory Dim: {config['mem_dim']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['lr']}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    success_count = 0
    failed_classes = []
    
    for class_name in classes:
        print(f"\n{mode.upper()}ING CLASS: {class_name}")
        print("-" * 40)
        
        # Build command arguments
        cmd_args = [
            'python', 'main.py',
            '--dataset', dataset,
            '--data_path', data_path,
            '--class_name', class_name,
            '--setting', 'single',
            '--mode', mode,
            '--gpu', str(gpu),
            '--seed', '42',
            '--backbone', config['backbone'],
            '--mem_dim', str(config['mem_dim']),
            '--batch_size', str(config['batch_size']),
            '--epochs', str(config['epochs']),
            '--lr', str(config['lr']),
            '--output_dir', output_dir,
            '--exp_name', f"mobilenet_{config_name}_{dataset}_{class_name}",
            '--use_spatial_memory',
            '--fusion_method', 'add'
        ]
        
        # Add test-specific arguments
        if mode == 'test':
            cmd_args.extend([
                '--save_images',
                '--checkpoint', f"{output_dir}/mobilenet_{config_name}_{dataset}_{class_name}/best_checkpoint.pth"
            ])
        
        print("Command:", ' '.join(cmd_args))
        print()
        
        # Execute command
        try:
            result = subprocess.run(cmd_args, check=True)
            print(f"✓ {class_name}: {mode.upper()} completed successfully!")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ {class_name}: {mode.upper()} failed with error: {e}")
            failed_classes.append(class_name)
        except FileNotFoundError:
            print("Error: main.py not found. Make sure you're in the DMIAD project root directory.")
            return False
        except KeyboardInterrupt:
            print(f"\n{mode.upper()} interrupted by user.")
            return False
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"MobileNet {mode.upper()} SUMMARY")
    print("=" * 60)
    print(f"Configuration: {config['description']}")
    print(f"Total classes: {len(classes)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_classes)}")
    
    if failed_classes:
        print(f"Failed classes: {', '.join(failed_classes)}")
    
    return len(failed_classes) == 0


def benchmark_mobilenet_variants(dataset, class_name, data_path, gpu=0):
    """
    Benchmark different MobileNet variants on a single class
    
    Args:
        dataset: Dataset name
        class_name: Single class to test
        data_path: Path to dataset
        gpu: GPU device ID
    """
    
    print(f"Benchmarking MobileNet variants on {dataset}/{class_name}")
    print("=" * 70)
    
    results = {}
    
    for config_name, config in MOBILENET_CONFIGS.items():
        print(f"\nTesting {config_name}: {config['description']}")
        print("-" * 50)
        
        timestamp = get_timestamp()
        output_dir = f'./benchmark/mobilenet_{config_name}_{dataset}_{class_name}_{timestamp}'
        
        # Run training for a few epochs to test
        custom_config = config.copy()
        custom_config['epochs'] = 10  # Quick test
        
        try:
            success = run_mobilenet_dmiad(
                dataset=dataset,
                classes=[class_name],
                config_name=config_name,
                data_path=data_path,
                mode='train',
                gpu=gpu,
                output_dir=output_dir,
                custom_config=custom_config
            )
            
            if success:
                results[config_name] = {
                    'status': 'success',
                    'config': config,
                    'output_dir': output_dir
                }
            else:
                results[config_name] = {
                    'status': 'failed',
                    'config': config
                }
                
        except Exception as e:
            results[config_name] = {
                'status': 'error',
                'error': str(e),
                'config': config
            }
    
    # Print benchmark summary
    print("\n" + "=" * 70)
    print("MOBILENET BENCHMARK SUMMARY")
    print("=" * 70)
    
    for config_name, result in results.items():
        status = result['status']
        config = result['config']
        
        print(f"{config_name:12} | {status:8} | {config['backbone']:20} | "
              f"Mem: {config['mem_dim']:4} | BS: {config['batch_size']:2}")
    
    return results


def compare_with_wideresnet(dataset, class_name, data_path, gpu=0):
    """
    Compare MobileNet performance with WideResNet baseline
    
    Args:
        dataset: Dataset name
        class_name: Single class to test
        data_path: Path to dataset  
        gpu: GPU device ID
    """
    
    print(f"Comparing MobileNet vs WideResNet on {dataset}/{class_name}")
    print("=" * 70)
    
    # Test configurations
    test_configs = {
        'WideResNet (Original)': {
            'backbone': 'wide_resnet',
            'mem_dim': 2000,
            'batch_size': 6,
            'epochs': 10,  # Quick test
            'lr': 1e-4
        },
        'MobileNetV2 (Balanced)': MOBILENET_CONFIGS['balanced'].copy(),
        'MobileNetV2 (Fast)': MOBILENET_CONFIGS['fast'].copy()
    }
    
    # Reduce epochs for quick comparison
    for config in test_configs.values():
        config['epochs'] = 10
    
    results = {}
    
    for config_name, config in test_configs.items():
        print(f"\nTesting {config_name}")
        print("-" * 50)
        
        timestamp = get_timestamp()
        output_dir = f'./comparison/{config_name.replace(" ", "_").lower()}_{dataset}_{class_name}_{timestamp}'
        
        # Build command arguments
        cmd_args = [
            'python', 'main.py',
            '--dataset', dataset,
            '--data_path', data_path,
            '--class_name', class_name,
            '--setting', 'single',
            '--mode', 'train',
            '--gpu', str(gpu),
            '--seed', '42',
            '--backbone', config['backbone'],
            '--mem_dim', str(config['mem_dim']),
            '--batch_size', str(config['batch_size']),
            '--epochs', str(config['epochs']),
            '--lr', str(config['lr']),
            '--output_dir', output_dir,
            '--exp_name', f"comparison_{config_name.replace(' ', '_').lower()}",
            '--use_spatial_memory',
            '--fusion_method', 'add'
        ]
        
        try:
            print("Running:", ' '.join(cmd_args[:10]) + "...")  # Show partial command
            result = subprocess.run(cmd_args, check=True, capture_output=True, text=True)
            
            results[config_name] = {
                'status': 'success',
                'config': config,
                'output_dir': output_dir
            }
            print(f"✓ {config_name}: Training completed")
            
        except subprocess.CalledProcessError as e:
            results[config_name] = {
                'status': 'failed', 
                'config': config,
                'error': str(e)
            }
            print(f"✗ {config_name}: Training failed")
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("MOBILENET vs WIDERESNET COMPARISON")
    print("=" * 70)
    
    for config_name, result in results.items():
        status = result['status']
        config = result['config']
        
        params = "~" + str(config['mem_dim'] * 2 + config.get('feature_dim', 256) * 1000 // 1000) + "M"
        
        print(f"{config_name:25} | {status:8} | "
              f"Mem: {config['mem_dim']:4} | BS: {config['batch_size']:2} | "
              f"Est.Params: {params}")
    
    return results


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='MobileNet DMIAD Runner')
    
    parser.add_argument('command', choices=['train', 'test', 'benchmark', 'compare', 'list'],
                       help='Command to execute')
    
    parser.add_argument('--dataset', choices=['mvtec', 'visa', 'mpdd'], 
                       default='mvtec', help='Dataset name')
    
    parser.add_argument('--class_name', type=str, 
                       help='Single class name (for benchmark/compare)')
    
    parser.add_argument('--classes', nargs='+', 
                       help='Multiple class names (for train/test)')
    
    parser.add_argument('--config', choices=list(MOBILENET_CONFIGS.keys()), 
                       default='balanced', help='MobileNet configuration')
    
    parser.add_argument('--data_path', type=str, 
                       help='Path to dataset')
    
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU device ID')
    
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Handle list command
    if args.command == 'list':
        print("Available MobileNet Configurations:")
        print("=" * 50)
        for name, config in MOBILENET_CONFIGS.items():
            print(f"{name:12} - {config['description']}")
            print(f"             Backbone: {config['backbone']}")
            print(f"             Memory: {config['mem_dim']}, Batch: {config['batch_size']}")
            print()
        
        print("Available Datasets:")
        print("=" * 50)
        for name, config in DATASET_CONFIGS.items():
            print(f"{name:8} - {len(config['classes'])} classes")
            print(f"         Path: {config['default_path']}")
            print()
        
        return
    
    # Get dataset configuration
    if args.dataset not in DATASET_CONFIGS:
        print(f"Error: Unknown dataset {args.dataset}")
        return
    
    dataset_config = DATASET_CONFIGS[args.dataset]
    
    # Determine data path
    data_path = args.data_path or dataset_config['default_path']
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset path does not exist: {data_path}")
        return
    
    # Determine classes to process
    if args.classes:
        classes = args.classes
    elif args.class_name:
        classes = [args.class_name]
    else:
        classes = dataset_config['classes']
    
    # Validate classes
    invalid_classes = [c for c in classes if c not in dataset_config['classes']]
    if invalid_classes:
        print(f"Error: Invalid classes for {args.dataset}: {invalid_classes}")
        print(f"Available classes: {dataset_config['classes']}")
        return
    
    # Execute command
    if args.command in ['train', 'test']:
        success = run_mobilenet_dmiad(
            dataset=args.dataset,
            classes=classes,
            config_name=args.config,
            data_path=data_path,
            mode=args.command,
            gpu=args.gpu,
            output_dir=args.output_dir
        )
        
        if not success:
            sys.exit(1)
    
    elif args.command == 'benchmark':
        if not args.class_name:
            print("Error: --class_name required for benchmark command")
            return
        
        results = benchmark_mobilenet_variants(
            dataset=args.dataset,
            class_name=args.class_name,
            data_path=data_path,
            gpu=args.gpu
        )
    
    elif args.command == 'compare':
        if not args.class_name:
            print("Error: --class_name required for compare command")
            return
        
        results = compare_with_wideresnet(
            dataset=args.dataset,
            class_name=args.class_name,
            data_path=data_path,
            gpu=args.gpu
        )


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("Error: main.py not found. Please run this script from the DMIAD project root directory.")
        sys.exit(1)
    
    main()