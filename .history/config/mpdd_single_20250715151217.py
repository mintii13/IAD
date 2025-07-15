import os
import sys

# Configuration for MPDD Single
config = {
    'datapath': r'D:\FPTU-sourse\Term5\ImageAnomalyDetection\CRAS\dataset\MPDD',
    'classes': ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'],
    'dataset_name': 'mpdd',
    'gpu': 0,
    'seed': 0,
    'test': 'ckpt',  # Change to 'test' for testing mode
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
    """Run MPDD single-class training/testing"""
    
    # Override test mode if provided
    if test_mode:
        config['test'] = test_mode
    
    # Build layers argument
    layers_args = ' '.join([f'-le {layer}' for layer in config['layers']])
    
    # Build classes argument
    classes_args = ' '.join([f'-d {cls}' for cls in config['classes']])
    
    # Build command
    cmd = f"""python main.py \\
    --gpu {config['gpu']} \\
    --seed {config['seed']} \\
    --test {config['test']} \\
  net \\
    -b {config['backbone']} \\
    {layers_args} \\
    --pretrain_embed_dimension {config['pretrain_embed_dimension']} \\
    --target_embed_dimension {config['target_embed_dimension']} \\
    --patchsize {config['patchsize']} \\
    --meta_epochs {config['meta_epochs']} \\
    --eval_epochs {config['eval_epochs']} \\
    --dsc_layers {config['dsc_layers']} \\
    --pre_proj {config['pre_proj']} \\
    --noise {config['noise']} \\
    --k {config['k']} \\
    --limit {config['limit']} \\
  dataset \\
    --setting {config['setting']} \\
    --batch_size {config['batch_size']} \\
    --resize {config['resize']} \\
    --imagesize {config['imagesize']} {classes_args} {config['dataset_name']} {config['datapath']}"""
    
    print("MPDD Single-Class Configuration")
    print("=" * 50)
    print(f"Dataset path: {config['datapath']}")
    print(f"Classes: {config['classes']}")
    print(f"Mode: {'Training' if config['test'] == 'ckpt' else 'Testing'}")
    print(f"Setting: {config['setting']}")
    print(f"Meta epochs: {config['meta_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print("=" * 50)
    
    # Clear screen (equivalent to 'clear' in bash)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Execute command
    print("Starting MPDD single-class training...")
    os.system(cmd)

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        test_mode = sys.argv[1]
        if test_mode in ['ckpt', 'test']:
            run_training(test_mode)
        else:
            print("Usage: python mpdd_single_config.py [ckpt|test]")
            print("  ckpt: Training mode (default)")
            print("  test: Testing mode")
    else:
        # Default: training mode
        run_training()