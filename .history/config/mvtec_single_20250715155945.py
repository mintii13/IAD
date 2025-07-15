import subprocess
import sys
import os

# Configuration for MVTec Single-Class
config = {
    'datapath': r'D:\FPTU-sourse\Term5\ImageAnomalyDetection\CRAS\dataset\MVTec',
    'classes': [
        'carpet', 'grid', 'leather', 'tile', 'wood', 
        'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
        'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
    ],
    'dataset_name': 'mvtec',
    'gpu': 0,
    'seed': 0,
    'test': 'ckpt',  # 'ckpt' for training, 'test' for testing
    'backbone': 'wideresnet50',
    'layers': ['layer2', 'layer3'],
    'pretrain_embed_dimension': 1536,
    'target_embed_dimension': 1536,
    'patchsize': 3,
    'meta_epochs': 640,  # Single-class uses 640 epochs
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

def run_mvtec_single(mode='train', selected_classes=None):
    """Run MVTec single-class training/testing"""
    
    # Set test mode
    if mode == 'test':
        config['test'] = 'test'
    else:
        config['test'] = 'ckpt'
    
    # Use selected classes or all classes
    if selected_classes:
        config['classes'] = selected_classes
    
    print("MVTec Single-Class Configuration")
    print("=" * 60)
    print(f"Dataset path: {config['datapath']}")
    print(f"Classes ({len(config['classes'])}): {config['classes']}")
    print(f"Mode: {'Training' if config['test'] == 'ckpt' else 'Testing'}")
    print(f"Setting: {config['setting']}")
    print(f"Meta epochs: {config['meta_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print("=" * 60)
    
    # Build command arguments
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
    
    print(f"Starting MVTec single-class {mode}ing...")
    print("Command:", ' '.join(cmd_args))
    print("-" * 60)
    
    # Set environment to suppress warnings
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'
    
    # Execute using subprocess
    try:
        result = subprocess.run(cmd_args, check=True, env=env)
        print(f"MVTec single-class {mode}ing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False
    except FileNotFoundError:
        print("Error: main.py not found. Make sure you're in the correct directory.")
        return False

def run_texture_classes():
    """Run only texture classes"""
    texture_classes = ['carpet', 'grid', 'leather', 'tile', 'wood']
    return run_mvtec_single('train', texture_classes)

def run_object_classes():
    """Run only object classes"""
    object_classes = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
                     'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    return run_mvtec_single('train', object_classes)

def run_single_class(class_name):
    """Run single class"""
    return run_mvtec_single('train', [class_name])

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("Error: main.py not found. Please run this script from the CRAS project root directory.")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        # Default: show options
        print("MVTec Single-Class Training Options:")
        print("=" * 50)
        print("  python mvtec_single_config.py train          # Train all 15 classes")
        print("  python mvtec_single_config.py test           # Test all 15 classes") 
        print("  python mvtec_single_config.py texture        # Train 5 texture classes")
        print("  python mvtec_single_config.py object         # Train 10 object classes")
        print("  python mvtec_single_config.py single <class> # Train single class")
        print("\nAvailable classes:")
        print("  Texture: carpet, grid, leather, tile, wood")
        print("  Object:  bottle, cable, capsule, hazelnut, metal_nut,")
        print("           pill, screw, toothbrush, transistor, zipper")
        print("\nExample: python mvtec_single_config.py single carpet")
        sys.exit(0)
        
    elif sys.argv[1] == 'train':
        success = run_mvtec_single('train')
        
    elif sys.argv[1] == 'test':
        success = run_mvtec_single('test')
        
    elif sys.argv[1] == 'texture':
        success = run_texture_classes()
        
    elif sys.argv[1] == 'object':
        success = run_object_classes()
        
    elif sys.argv[1] == 'single':
        if len(sys.argv) < 3:
            print("Error: Please specify class name for single mode")
            print("Example: python mvtec_single_config.py single carpet")
            print("Available classes: carpet, grid, leather, tile, wood, bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper")
            sys.exit(1)
        class_name = sys.argv[2]
        if class_name not in config['classes']:
            print(f"Error: '{class_name}' is not a valid MVTec class")
            print(f"Available classes: {', '.join(config['classes'])}")
            sys.exit(1)
        success = run_single_class(class_name)
        
    else:
        print("Invalid option. Use: train, test, texture, object, or single <class_name>")
        sys.exit(1)
    
    if not success:
        sys.exit(1)