import subprocess
import sys
import os
from datetime import datetime
timestamp = datetime.now().strftime("%d%m%Y_%H%M")

# Configuration for MVTec DMIAD
config = {
    'datapath': r'D:\FPTU-sourse\Term5\ImageAnomalyDetection\DMIAD\dataset\MVTec',
    'classes': [
        'carpet', 'grid', 'leather', 'tile', 'wood',  # Texture classes
        'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',  # Object classes
        'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
    ],
    'texture_classes': ['carpet', 'grid', 'leather', 'tile', 'wood'],
    'object_classes': ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
                      'pill', 'screw', 'toothbrush', 'transistor', 'zipper'],
    'dataset_name': 'mvtec',
    'gpu': 0,
    'seed': 0,
    'mode': 'train',  # 'train' or 'test'
    
    # Backbone options (just the names, keep your original config)
    'backbone': 'wide_resnet',  # Default backbone
    'available_backbones': [
        'wide_resnet', 
        'mobilenet_v2', 
        'mobilenet_v2_0.5', 
        'mobilenet_v2_0.75',
        'mobilenet_v2_1.4', 
        'mobilenet_v2_0.35',
        'mobilenet_v3'
    ],
    
    'mem_dim': 100,  # Memory dimension (from your target_embed_dimension)
    'epochs': 640,    # From your meta_epochs
    'batch_size': 6,  # From your script
    'lr': 1e-4,
    'setting': 'single',  # Single-class setting
    'resize': 329,        # From your script
    'imagesize': 288,     # From your script
    'use_spatial_memory': True,
    'fusion_method': 'add',
    'output_dir': f'./results/mvtec_results_{timestamp}'
}

def run_dmiad_mvtec(mode='train', selected_classes=None, backbone=None):
    """Run DMIAD training/testing on MVTec dataset"""
    
    # Use selected classes or all classes
    if selected_classes:
        classes_to_run = selected_classes
    else:
        classes_to_run = config['classes']
    
    # Use specified backbone or default
    if backbone is None:
        backbone = config['backbone']
    
    print("DMIAD MVTec Configuration")
    print("=" * 60)
    print(f"Dataset path: {config['datapath']}")
    print(f"Classes ({len(classes_to_run)}): {classes_to_run}")
    print(f"Mode: {mode.upper()}")
    print(f"Backbone: {backbone}")
    print(f"Setting: {config['setting']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Memory dimension: {config['mem_dim']}")
    print(f"Image size: {config['imagesize']}")
    print("=" * 60)
    
    # Run for each class
    success_count = 0
    failed_classes = []
    
    for class_name in classes_to_run:
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
            '--backbone', backbone,
            '--mem_dim', str(config['mem_dim']),
            '--batch_size', str(config['batch_size']),
            '--epochs', str(config['epochs']),
            '--lr', str(config['lr']),
            '--output_dir', config['output_dir'],
            '--exp_name', f"dmiad_mvtec_{backbone}_{class_name}",
            '--use_spatial_memory',
            '--fusion_method', config['fusion_method']
        ]
        
        # Add test-specific arguments
        if mode == 'test':
            cmd_args.extend([
                '--save_images',
                '--checkpoint', f"{config['output_dir']}/dmiad_mvtec_{backbone}_{class_name}/best_checkpoint.pth"
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
    print(f"MVTec {mode.upper()} SUMMARY")
    print("=" * 60)
    print(f"Backbone: {backbone}")
    print(f"Total classes: {len(classes_to_run)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_classes)}")
    
    if failed_classes:
        print(f"Failed classes: {', '.join(failed_classes)}")
    
    return len(failed_classes) == 0

def run_texture_classes(mode='train', backbone=None):
    """Run only texture classes"""
    print("Running TEXTURE classes...")
    return run_dmiad_mvtec(mode, config['texture_classes'], backbone)

def run_object_classes(mode='train', backbone=None):
    """Run only object classes"""
    print("Running OBJECT classes...")
    return run_dmiad_mvtec(mode, config['object_classes'], backbone)

def run_single_class(class_name, mode='train', backbone=None):
    """Run single MVTec class"""
    if class_name not in config['classes']:
        print(f"Error: '{class_name}' is not a valid MVTec class")
        print(f"Available classes: {', '.join(config['classes'])}")
        return False
    
    return run_dmiad_mvtec(mode, [class_name], backbone)

def check_data_path():
    """Check if MVTec dataset path exists"""
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
        # Remove missing classes from all class lists
        config['classes'] = [c for c in config['classes'] if c not in missing_classes]
        config['texture_classes'] = [c for c in config['texture_classes'] if c not in missing_classes]
        config['object_classes'] = [c for c in config['object_classes'] if c not in missing_classes]
    
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
        print("DMIAD MVTec Training/Testing Options:")
        print("=" * 50)
        print("  python mvtec_runner.py train [backbone]                    # Train all 15 classes")
        print("  python mvtec_runner.py test [backbone]                     # Test all 15 classes")
        print("  python mvtec_runner.py texture [backbone] [mode]           # Train/test 5 texture classes")
        print("  python mvtec_runner.py object [backbone] [mode]            # Train/test 10 object classes")
        print("  python mvtec_runner.py single <class> [backbone] [mode]    # Train/test single class")
        print()
        print("Available backbones:")
        print("  ", ', '.join(config['available_backbones']))
        print(f"  Default: {config['backbone']}")
        print()
        print("Available MVTec classes:")
        print("  Texture (5):", ', '.join(config['texture_classes']))
        print("  Object (10): ", ', '.join(config['object_classes']))
        print()
        print("Examples:")
        print("  python mvtec_runner.py train")
        print("  python mvtec_runner.py train mobilenet_v2")
        print("  python mvtec_runner.py texture mobilenet_v2_0.5 test")
        print("  python mvtec_runner.py single carpet")
        print("  python mvtec_runner.py single bottle mobilenet_v2 test")
        sys.exit(0)
    
    # Parse arguments
    command = sys.argv[1]
    
    # Extract backbone and mode from arguments
    backbone = None
    mode = 'train'
    
    if command == 'train':
        if len(sys.argv) > 2 and sys.argv[2] in config['available_backbones']:
            backbone = sys.argv[2]
        success = run_dmiad_mvtec('train', backbone=backbone)
        
    elif command == 'test':
        if len(sys.argv) > 2 and sys.argv[2] in config['available_backbones']:
            backbone = sys.argv[2]
        success = run_dmiad_mvtec('test', backbone=backbone)
        
    elif command == 'texture':
        # Parse backbone and mode for texture
        if len(sys.argv) > 2 and sys.argv[2] in config['available_backbones']:
            backbone = sys.argv[2]
            if len(sys.argv) > 3 and sys.argv[3] in ['train', 'test']:
                mode = sys.argv[3]
        elif len(sys.argv) > 2 and sys.argv[2] in ['train', 'test']:
            mode = sys.argv[2]
        
        success = run_texture_classes(mode, backbone)
        
    elif command == 'object':
        # Parse backbone and mode for object
        if len(sys.argv) > 2 and sys.argv[2] in config['available_backbones']:
            backbone = sys.argv[2]
            if len(sys.argv) > 3 and sys.argv[3] in ['train', 'test']:
                mode = sys.argv[3]
        elif len(sys.argv) > 2 and sys.argv[2] in ['train', 'test']:
            mode = sys.argv[2]
        
        success = run_object_classes(mode, backbone)
        
    elif command == 'single':
        if len(sys.argv) < 3:
            print("Error: Please specify class name for single mode")
            print("Example: python mvtec_runner.py single carpet")
            print(f"Available classes: {', '.join(config['classes'])}")
            sys.exit(1)
        
        class_name = sys.argv[2]
        
        # Parse backbone and mode for single
        if len(sys.argv) > 3 and sys.argv[3] in config['available_backbones']:
            backbone = sys.argv[3]
            if len(sys.argv) > 4 and sys.argv[4] in ['train', 'test']:
                mode = sys.argv[4]
        elif len(sys.argv) > 3 and sys.argv[3] in ['train', 'test']:
            mode = sys.argv[3]
        
        success = run_single_class(class_name, mode, backbone)
        
    else:
        print("Invalid option. Use: train, test, texture, object, or single <class_name>")
        sys.exit(1)
    
    if not success:
        sys.exit(1)