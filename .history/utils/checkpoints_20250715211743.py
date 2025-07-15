"""
Checkpoint utilities for saving and loading model states
"""

import os
import torch
import logging


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