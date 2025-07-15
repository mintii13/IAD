"""
Checkpoint utilities for saving and loading model states
"""

import os
import torch
import logging
import time
import shutil


def save_checkpoint(state, filepath):
    """
    Save model checkpoint
    
    Args:
        state: Dict containing model state, optimizer state, etc.
        filepath: Path to save checkpoint
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save checkpoint
        torch.save(state, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
        
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {filepath}: {str(e)}")
        raise


def load_checkpoint(filepath, device='cpu', strict=True):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load checkpoint to
        strict: Whether to strictly enforce state dict keys
    
    Returns:
        Dict containing loaded checkpoint data
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=device)
        logging.info(f"Checkpoint loaded from {filepath}")
        
        return checkpoint
        
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {filepath}: {str(e)}")
        raise


def load_model_weights(model, checkpoint_path, device='cpu', strict=True):
    """
    Load only model weights from checkpoint
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load weights to
        strict: Whether to strictly enforce state dict keys
    
    Returns:
        model: Model with loaded weights
    """
    try:
        checkpoint = load_checkpoint(checkpoint_path, device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=strict)
        else:
            # Assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint, strict=strict)
        
        logging.info(f"Model weights loaded from {checkpoint_path}")
        return model
        
    except Exception as e:
        logging.error(f"Failed to load model weights from {checkpoint_path}: {str(e)}")
        raise


def create_checkpoint(model, optimizer, scheduler, epoch, metrics, config, best_score=None, global_step=0):
    """
    Create checkpoint dictionary
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        metrics: Current metrics dict
        config: Configuration object
        best_score: Best validation score so far
        global_step: Global training step
    
    Returns:
        Dict: Checkpoint dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'global_step': global_step,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if best_score is not None:
        checkpoint['best_score'] = best_score
    
    return checkpoint


def resume_training(model, optimizer, scheduler, checkpoint_path, device='cpu'):
    """
    Resume training from checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
    
    Returns:
        Tuple: (model, optimizer, scheduler, start_epoch, best_score, global_step)
    """
    try:
        checkpoint = load_checkpoint(checkpoint_path, device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Extract training state
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_score = checkpoint.get('best_score', 0.0)
        global_step = checkpoint.get('global_step', 0)
        
        logging.info(f"Resumed training from epoch {start_epoch}")
        
        return model, optimizer, scheduler, start_epoch, best_score, global_step
        
    except Exception as e:
        logging.error(f"Failed to resume training from {checkpoint_path}: {str(e)}")
        raise


def cleanup_checkpoints(checkpoint_dir, keep_last_n=5, keep_best=True):
    """
    Clean up old checkpoints, keeping only the most recent ones
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
    """
    try:
        if not os.path.exists(checkpoint_dir):
            return
        
        # Get all checkpoint files
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pth') and 'checkpoint_epoch_' in file:
                epoch_num = int(file.split('_')[-1].split('.')[0])
                checkpoint_files.append((epoch_num, file))
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: x[0])
        
        # Files to keep
        files_to_keep = set()
        
        # Keep last N checkpoints
        if len(checkpoint_files) > keep_last_n:
            for epoch_num, file in checkpoint_files[-keep_last_n:]:
                files_to_keep.add(file)
        else:
            for epoch_num, file in checkpoint_files:
                files_to_keep.add(file)
        
        # Always keep latest and best checkpoints
        files_to_keep.add('latest_checkpoint.pth')
        if keep_best:
            files_to_keep.add('best_checkpoint.pth')
        
        # Remove old checkpoints
        for epoch_num, file in checkpoint_files:
            if file not in files_to_keep:
                file_path = os.path.join(checkpoint_dir, file)
                os.remove(file_path)
                logging.info(f"Removed old checkpoint: {file}")
                
    except Exception as e:
        logging.warning(f"Failed to cleanup checkpoints: {str(e)}")


def get_checkpoint_info(checkpoint_path):
    """
    Get information about a checkpoint without fully loading it
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Dict: Checkpoint information
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'metrics': checkpoint.get('metrics', {}),
            'best_score': checkpoint.get('best_score', 'Unknown'),
            'global_step': checkpoint.get('global_step', 'Unknown'),
        }
        
        # Add model parameter count if available
        if 'model_state_dict' in checkpoint:
            total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
            info['total_parameters'] = total_params
        
        return info
        
    except Exception as e:
        logging.error(f"Failed to get checkpoint info from {checkpoint_path}: {str(e)}")
        return {}


def convert_checkpoint_format(old_checkpoint_path, new_checkpoint_path, format_type='dmiad'):
    """
    Convert checkpoint from one format to another
    
    Args:
        old_checkpoint_path: Path to old checkpoint
        new_checkpoint_path: Path to save new checkpoint
        format_type: Target format type
    """
    try:
        old_checkpoint = load_checkpoint(old_checkpoint_path)
        
        if format_type == 'dmiad':
            # Convert to DMIAD format
            new_checkpoint = {
                'epoch': old_checkpoint.get('epoch', 0),
                'model_state_dict': old_checkpoint.get('model_state_dict', old_checkpoint.get('state_dict', {})),
                'optimizer_state_dict': old_checkpoint.get('optimizer_state_dict', {}),
                'metrics': old_checkpoint.get('metrics', {}),
                'best_score': old_checkpoint.get('best_score', 0.0),
                'global_step': old_checkpoint.get('global_step', 0),
            }
            
            if 'scheduler_state_dict' in old_checkpoint:
                new_checkpoint['scheduler_state_dict'] = old_checkpoint['scheduler_state_dict']
        
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        save_checkpoint(new_checkpoint, new_checkpoint_path)
        logging.info(f"Checkpoint converted from {old_checkpoint_path} to {new_checkpoint_path}")
        
    except Exception as e:
        logging.error(f"Failed to convert checkpoint: {str(e)}")
        raise


def validate_checkpoint(checkpoint_path):
    """
    Validate checkpoint file integrity
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Check required keys
        required_keys = ['model_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                logging.error(f"Missing required key in checkpoint: {key}")
                return False
        
        # Check if model state dict is not empty
        if not checkpoint['model_state_dict']:
            logging.error("Model state dict is empty")
            return False
        
        logging.info(f"Checkpoint validation passed: {checkpoint_path}")
        return True
        
    except Exception as e:
        logging.error(f"Checkpoint validation failed: {str(e)}")
        return False


def backup_checkpoint(checkpoint_path, backup_dir=None):
    """
    Create backup of checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint file
        backup_dir: Directory to save backup (optional)
    
    Returns:
        str: Path to backup file
    """
    try:
        if backup_dir is None:
            backup_dir = os.path.join(os.path.dirname(checkpoint_path), 'backups')
        
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup filename with timestamp
        filename = os.path.basename(checkpoint_path)
        name, ext = os.path.splitext(filename)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{name}_backup_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy checkpoint to backup location
        shutil.copy2(checkpoint_path, backup_path)
        
        logging.info(f"Checkpoint backed up to: {backup_path}")
        return backup_path
        
    except Exception as e:
        logging.error(f"Failed to backup checkpoint: {str(e)}")
        raise


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import torch.nn as nn
    import torch.optim as optim
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Create dummy metrics
    metrics = {
        'train_loss': 0.5,
        'val_loss': 0.6,
        'accuracy': 0.85
    }
    
    # Test checkpoint creation and saving
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        
        # Create and save checkpoint
        checkpoint = create_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=10,
            metrics=metrics,
            config={'test': True},
            best_score=0.9,
            global_step=1000
        )
        
        save_checkpoint(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}")
        
        # Test checkpoint loading
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        print(f"✓ Checkpoint loaded, epoch: {loaded_checkpoint['epoch']}")
        
        # Test checkpoint info
        info = get_checkpoint_info(checkpoint_path)
        print(f"✓ Checkpoint info: {info}")
        
        # Test model weights loading
        new_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        new_model = load_model_weights(new_model, checkpoint_path)
        print("✓ Model weights loaded successfully")
        
        # Test checkpoint validation
        is_valid = validate_checkpoint(checkpoint_path)
        print(f"✓ Checkpoint validation: {is_valid}")
        
        # Test backup
        backup_path = backup_checkpoint(checkpoint_path, temp_dir)
        print(f"✓ Checkpoint backed up to: {backup_path}")
        
        # Test resume training
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=10, gamma=0.1)
        
        _, _, _, start_epoch, best_score, global_step = resume_training(
            new_model, new_optimizer, new_scheduler, checkpoint_path
        )
        print(f"✓ Resume training: epoch {start_epoch}, best_score {best_score}")
    
    print("\nCheckpoint utilities test completed!")