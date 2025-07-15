"""
Training script for DMIAD
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from datasets.mvtec import get_dataloaders, DatasetSplit
from models.dmiad_model import build_dmiad_model
from models.losses.combined_loss import CombinedLoss
from utils.metrics import compute_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import get_logger
from utils.visualization import save_visualization


class Trainer:
    """Training class for DMIAD"""
    
    def __init__(self, config, device, exp_dir):
        self.config = config
        self.device = device
        self.exp_dir = exp_dir
        
        # Setup logging
        self.logger = get_logger(exp_dir)
        self.writer = SummaryWriter(os.path.join(exp_dir, 'tensorboard'))
        
        # Create model
        self.model = build_dmiad_model(config).to(device)
        self.logger.info(f"Model created: {self.model.__class__.__name__}")
        
        # Create dataloaders
        self.dataloaders = get_dataloaders(config)
        self.logger.info(f"Train samples: {len(self.dataloaders['train'].dataset)}")
        self.logger.info(f"Test samples: {len(self.dataloaders['test'].dataset)}")
        
        # Create loss function
        self.criterion = CombinedLoss(config).to(device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.start_epoch = 0
        self.best_score = 0.0
        self.global_step = 0
        
        # Load checkpoint if exists
        self._load_checkpoint_if_exists()
    
    def _create_optimizer(self):
        """Create optimizer"""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.TRAIN.LR,
            weight_decay=self.config.TRAIN.WEIGHT_DECAY
        )
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.TRAIN.SCHEDULER == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.TRAIN.EPOCHS,
                eta_min=self.config.TRAIN.LR * 0.01
            )
        elif self.config.TRAIN.SCHEDULER == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.TRAIN.EPOCHS // 3,
                gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _load_checkpoint_if_exists(self):
        """Load checkpoint if exists"""
        checkpoint_path = os.path.join(self.exp_dir, 'latest_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_score = checkpoint.get('best_score', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
            
            self.logger.info(f"Resumed from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = len(self.dataloaders['train'])
        
        progress_bar = tqdm(
            self.dataloaders['train'], 
            desc=f'Epoch {epoch}/{self.config.TRAIN.EPOCHS}',
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            reconstructed = outputs['reconstructed']
            memory_results = outputs['memory_results']
            
            # Compute loss
            loss_dict = self.criterion(
                images, reconstructed, memory_results, labels
            )
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            self.global_step += 1
            
            # Log to tensorboard
            if self.global_step % 50 == 0:
                self._log_training_step(loss_dict, epoch, batch_idx, num_batches)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        
        # Log epoch metrics
        self.writer.add_scalar('Train/EpochLoss', avg_epoch_loss, epoch)
        self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
        
        self.logger.info(f'Epoch {epoch}: Average Loss = {avg_epoch_loss:.4f}')
        
        return avg_epoch_loss
    
    def _log_training_step(self, loss_dict, epoch, batch_idx, num_batches):
        """Log training step to tensorboard"""
        for loss_name, loss_value in loss_dict.items():
            self.writer.add_scalar(f'Train/{loss_name}', loss_value.item(), self.global_step)
        
        # Log memory statistics if available
        memory_items = self.model.get_memory_items()
        for mem_name, mem_tensor in memory_items.items():
            # Log memory norm
            mem_norm = torch.norm(mem_tensor).item()
            self.writer.add_scalar(f'Memory/{mem_name}_norm', mem_norm, self.global_step)
    
    def validate_epoch(self, epoch):
        """Validate one epoch"""
        self.model.eval()
        
        val_loss = 0.0
        all_scores = []
        all_labels = []
        all_pixel_scores = []
        all_pixel_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloaders['test'], desc='Validation', leave=False)):
                # Move data to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                reconstructed = outputs['reconstructed']
                memory_results = outputs['memory_results']
                
                # Compute loss
                loss_dict = self.criterion(images, reconstructed, memory_results, labels)
                val_loss += loss_dict['total_loss'].item()
                
                # Compute anomaly scores
                anomaly_scores = self.model.compute_anomaly_score(images, reconstructed)
                
                # Collect scores and labels
                all_scores.extend(anomaly_scores['image_scores'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Collect pixel-level scores (flatten spatial dimensions)
                pixel_scores = anomaly_scores['pixel_scores'].cpu().numpy()
                pixel_labels = masks.cpu().numpy()
                
                for i in range(pixel_scores.shape[0]):
                    all_pixel_scores.extend(pixel_scores[i].flatten())
                    all_pixel_labels.extend(pixel_labels[i].flatten())
        
        # Compute metrics
        metrics = compute_metrics(
            np.array(all_scores),
            np.array(all_labels),
            np.array(all_pixel_scores),
            np.array(all_pixel_labels)
        )
        
        # Average validation loss
        avg_val_loss = val_loss / len(self.dataloaders['test'])
        
        # Log metrics
        self.writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        self.writer.add_scalar('Val/ImageAUROC', metrics['image_auroc'], epoch)
        self.writer.add_scalar('Val/PixelAUROC', metrics['pixel_auroc'], epoch)
        
        self.logger.info(f'Validation - Loss: {avg_val_loss:.4f}, '
                        f'Image AUROC: {metrics["image_auroc"]:.4f}, '
                        f'Pixel AUROC: {metrics["pixel_auroc"]:.4f}')
        
        return metrics, avg_val_loss
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
            'global_step': self.global_step,
            'config': self.config,
            'metrics': metrics,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        save_checkpoint(checkpoint, os.path.join(self.exp_dir, 'latest_checkpoint.pth'))
        
        # Save best checkpoint
        if is_best:
            save_checkpoint(checkpoint, os.path.join(self.exp_dir, 'best_checkpoint.pth'))
            self.logger.info(f'New best model saved at epoch {epoch}')
        
        # Save periodic checkpoint
        if epoch % self.config.SAVE_FREQ == 0:
            save_checkpoint(checkpoint, os.path.join(self.exp_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Training from epoch {self.start_epoch} to {self.config.TRAIN.EPOCHS}")
        
        for epoch in range(self.start_epoch, self.config.TRAIN.EPOCHS):
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Validation phase
            if epoch % self.config.EVAL_FREQ == 0:
                metrics, val_loss = self.validate_epoch(epoch)
                
                # Check if best model
                current_score = metrics['image_auroc'] + metrics['pixel_auroc']
                is_best = current_score > self.best_score
                if is_best:
                    self.best_score = current_score
                
                # Save checkpoint
                self.save_checkpoint(epoch, metrics, is_best)
                
                # Save visualization
                if epoch % (self.config.EVAL_FREQ * 2) == 0:
                    self._save_visualizations(epoch)
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def _save_visualizations(self, epoch):
        """Save training visualizations"""
        # TODO: Implement visualization saving
        # - Sample reconstructions
        # - Attention maps
        # - Memory patterns
        pass


def train_model(config, device, exp_dir):
    """Main training function"""
    
    # Create trainer
    trainer = Trainer(config, device, exp_dir)
    
    # Start training
    trainer.train()
    
    return trainer


# Example usage
if __name__ == "__main__":
    from config.base_config import get_config
    import argparse
    
    # Parse arguments
    args = argparse.Namespace()
    args.dataset = 'mvtec'
    args.data_path = '/path/to/mvtec'
    args.class_name = 'bottle'
    args.setting = 'single'
    args.batch_size = 8
    args.gpu = 0
    args.mem_dim = 2000
    args.use_spatial_memory = True
    args.fusion_method = 'add'
    args.backbone = 'wide_resnet'
    args.epochs = 100
    args.lr = 1e-4
    args.output_dir = './results'
    args.exp_name = 'test_training'
    
    # Get config
    config = get_config(args)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment directory
    exp_dir = os.path.join(config.OUTPUT_DIR, config.EXP_NAME)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Start training
    trainer = train_model(config, device, exp_dir)