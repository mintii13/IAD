import subprocess
import sys
import os

# Configuration for MPDD DMIAD
config = {
    'datapath': r'D:\FPTU-sourse\Term5\ImageAnomalyDetection\DMIAD\dataset\MPDD',
    'classes': ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'],
    'dataset_name': 'mpdd',
    'gpu': 0,
    'seed': 0,
    'mode': 'train',  # 'train' or 'test'
    'backbone': 'wide_resnet',
    'mem_dim': 1536,  # Memory dimension (from your target_embed_dimension)
    'epochs': 640,    # From your meta_epochs
    'batch_size': 8,  # From your script
    'lr': 1e-4,
    'setting': 'single',  # Single-class setting
    'resize': 329,        # From your script
    'imagesize': 288,     # From your script
    'use_spatial_memory': True,
    'fusion_method': 'add',
    'output_dir': './mpdd_results'
}

def run_dmiad_mpdd(mode='train', selected_classes=None):
    """Run DMIAD training/testing on MPDD dataset"""
    
    # Use selected classes or all classes
    if selected_classes:
        config['classes'] = selected_classes
    
    # Set mode
    config['mode'] = mode
    
    print("DMIAD MPDD Configuration")
    print("=" * 60)
    print(f"Dataset path: {config['datapath']}")
    print(f"Classes ({len(config['classes'])}): {config['classes']}")
    print(f"Mode: {mode.upper()}")
    print(f"Setting: {config['setting']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Memory dimension: {config['mem_dim']}")
    print(f"Image size: {config['imagesize']}")
    print("=" * 60)
    
    # Run for each class
    success_count = 0
    failed_classes = []
    
    for class_name in config['classes']:
        print(f"\n{mode.upper()}ING CLASS: {class_name}")
        print("-" * 40)
        
        # Build command arguments
        cmd_args = [
            'python', 'main.py',
            '--dataset', config['dataset_name'],
            '--data_path', config['datapath'],
            '--class_name', class_name,
            '--setting', config['setting'],
            '--mode', config['mode'],
            '--gpu', str(config['gpu']),
            '--seed', str(config['seed']),
            '--backbone', config['backbone'],
            '--mem_dim', str(config['mem_dim']),
            '--batch_size', str(config['batch_size']),
            '--epochs', str(config['epochs']),
            '--lr', str(config['lr']),
            '--output_dir', config['output_dir'],
            '--exp_name', f"dmiad_mpdd_{class_name}"
        ]
        
        # Add spatial memory flag
        if config['use_spatial_memory']:
            cmd_args.append('--use_spatial_memory')
        
        # Add fusion method
        cmd_args.extend(['--fusion_method', config['fusion_method']])
        
        # Add test-specific arguments
        if mode == 'test':
            cmd_args.extend([
                '--save_images',
                '--checkpoint', f"{config['output_dir']}/dmiad_mpdd_{class_name}/best_checkpoint.pth"
            ])
        
        print("Command:", ' '.join(cmd_args))
        print()
        
        # Execute using subprocess
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
    print(f"MPDD {mode.upper()} SUMMARY")
    print("=" * 60)
    print(f"Total classes: {len(config['classes'])}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_classes)}")
    
    if failed_classes:
        print(f"Failed classes: {', '.join(failed_classes)}")
    
    return len(failed_classes) == 0

def run_single_class(class_name, mode='train'):
    """Run single MPDD class"""
    if class_name not in config['classes']:
        print(f"Error: '{class_name}' is not a valid MPDD class")
        print(f"Available classes: {', '.join(config['classes'])}")
        return False
    
    return run_dmiad_mpdd(mode, [class_name])

def check_data_path():
    """Check if MPDD dataset path exists"""
    if not os.path.exists(config['datapath']):
        print(f"Error: Dataset path does not exist: {config['datapath']}")
        print("Please update the 'datapath' in the config section.")
        return False
    
    # Check if classes exist
    missing_classes = []
    for class_name in config['classes']:
        class_path = os.path.join(config['datapath'], class_name)
        if not os.path.exists(class_path):
            missing_classes.append(class_name)
    
    if missing_classes:
        print(f"Warning: Missing class directories: {missing_classes}")
        print("Available classes will be processed.")
        config['classes'] = [c for c in config['classes'] if c not in missing_classes]
    
    return True

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("Error: main.py not found. Please run this script from the DMIAD project root directory.")
        sys.exit(1)
    
    # Check dataset path
    if not check_data_path():
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        # Default: show options
        print("DMIAD MPDD Training/Testing Options:")
        print("=" * 50)
        print("  python mpdd_runner.py train                    # Train all 6 classes")
        print("  python mpdd_runner.py test                     # Test all 6 classes")
        print("  python mpdd_runner.py single <class> [mode]    # Train/test single class")
        print()
        print("Available MPDD classes:")
        print("  ", ', '.join(config['classes']))
        print()
        print("Examples:")
        print("  python mpdd_runner.py train")
        print("  python mpdd_runner.py test") 
        print("  python mpdd_runner.py single bracket_black")
        print("  python mpdd_runner.py single bracket_black test")
        sys.exit(0)
        
    elif sys.argv[1] == 'train':
        success = run_dmiad_mpdd('train')
        
    elif sys.argv[1] == 'test':
        success = run_dmiad_mpdd('test')
        
    elif sys.argv[1] == 'single':
        if len(sys.argv) < 3:
            print("Error: Please specify class name for single mode")
            print("Example: python mpdd_runner.py single bracket_black")
            print(f"Available classes: {', '.join(config['classes'])}")
            sys.exit(1)
        
        class_name = sys.argv[2]
        mode = sys.argv[3] if len(sys.argv) > 3 else 'train'
        
        if mode not in ['train', 'test']:
            print("Error: Mode must be 'train' or 'test'")
            sys.exit(1)
            
        success = run_single_class(class_name, mode)
        
    else:
        print("Invalid option. Use: train, test, or single <class_name> [train/test]")
        sys.exit(1)
    
    if not success:
        sys.exit(1)