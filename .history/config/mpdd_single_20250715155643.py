import subprocess
import sys
import os

# Configuration for MPDD Single
config = {
    'datapath': r'D:\FPTU-sourse\Term5\ImageAnomalyDetection\CRAS\dataset\MPDD',
    'classes': ['bracket_black', 'bracket_brown', 'bracket_white'],
    'dataset_name': 'mpdd',
    'gpu': 0,
    'seed': 0,
    'test': 'test',  # Change to 'test' for testing mode
    'backbone': 'wideresnet50',
    'layers': ['layer2', 'layer3'],
    'pretrain_embed_dimension': 1536,
    'target_embed_dimension': 1536,
    'patchsize': 3,
    'meta_epochs': 640,  # Single-class setting uses 640 epochs
    'eval_epochs': 1,
    'dsc_layers': 3,
    'pre_proj': 1,
    'noise': 0.015,
    'k': 0.3,
    'limit': -1,
    'setting': 'single',  # Single-class setting
    'batch_size': 8,      # Single-class uses smaller batch size
    'resize': 329,
    'imagesize': 288
}

def run_training(test_mode=None):
    """Run MPDD single-class training/testing using subprocess"""
    
    # Override test mode if provided
    if test_mode:
        config['test'] = test_mode
    
    print("MPDD Single-Class Configuration")
    print("=" * 50)
    print(f"Dataset path: {config['datapath']}")
    print(f"Classes: {config['classes']}")
    print(f"Mode: {'Training' if config['test'] == 'ckpt' else 'Testing'}")
    print(f"Setting: {config['setting']}")
    print(f"Meta epochs: {config['meta_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print("=" * 50)
    
    # Build command arguments as list
    cmd_args = [
        'python', 'main.py',
        '--gpu', str(config['gpu']),
        '--seed', str(config['seed']),
        '--test', config['test'],
        'net',
        '-b', config['backbone']
    ]
    
    # Add layers
    for layer in config['layers']:
        cmd_args.extend(['-le', layer])
    
    # Add network parameters
    cmd_args.extend([
        '--pretrain_embed_dimension', str(config['pretrain_embed_dimension']),
        '--target_embed_dimension', str(config['target_embed_dimension']),
        '--patchsize', str(config['patchsize']),
        '--meta_epochs', str(config['meta_epochs']),
        '--eval_epochs', str(config['eval_epochs']),
        '--dsc_layers', str(config['dsc_layers']),
        '--pre_proj', str(config['pre_proj']),
        '--noise', str(config['noise']),
        '--k', str(config['k']),
        '--limit', str(config['limit']),
        'dataset',
        '--setting', config['setting'],
        '--batch_size', str(config['batch_size']),
        '--resize', str(config['resize']),
        '--imagesize', str(config['imagesize'])
    ])
    
    # Add classes
    for cls in config['classes']:
        cmd_args.extend(['-d', cls])
    
    # Add dataset name and path
    cmd_args.extend([config['dataset_name'], config['datapath']])
    
    print("Starting MPDD single-class training...")
    print("Command:", ' '.join(cmd_args))
    print("-" * 50)
    
    # Execute using subprocess
    try:
        result = subprocess.run(cmd_args, check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False
    except FileNotFoundError:
        print("Error: main.py not found. Make sure you're in the correct directory.")
        return False
    
    return True

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("Error: main.py not found. Please run this script from the CRAS project root directory.")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        test_mode = sys.argv[1]
        if test_mode in ['ckpt', 'test']:
            success = run_training(test_mode)
        else:
            print("Usage: python mpdd_direct_runner.py [ckpt|test]")
            print("  ckpt: Training mode (default)")
            print("  test: Testing mode")
            sys.exit(1)
    else:
        # Default: training mode
        success = run_training()
    
    if not success:
        sys.exit(1)