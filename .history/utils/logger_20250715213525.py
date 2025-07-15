"""
Logging utilities for DMIAD
"""

import os
import sys
import logging
import time
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO, format_str=None):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_str: Custom format string (optional)
    
    Returns:
        Logger instance
    """
    if format_str is None:
        format_str = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    
    formatter = logging.Formatter(format_str)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplication
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(exp_dir, filename='train.log', level=logging.INFO):
    """
    Get logger for experiment directory
    
    Args:
        exp_dir: Experiment directory
        filename: Log filename
        level: Logging level
    
    Returns:
        Logger instance
    """
    log_file = os.path.join(exp_dir, filename)
    logger_name = f"DMIAD_{os.path.basename(exp_dir)}"
    
    return setup_logger(logger_name, log_file, level)


class MetricsLogger:
    """Logger for training metrics"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.metrics_history = []
        
        # Create log directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize log file with header
        with open(log_file, 'w') as f:
            f.write("timestamp,epoch,phase,loss,lr,image_auroc,pixel_auroc\n")
    
    def log_metrics(self, epoch, phase, metrics_dict):
        """
        Log metrics to file
        
        Args:
            epoch: Current epoch
            phase: Training phase ('train' or 'val')
            metrics_dict: Dictionary of metrics
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Extract common metrics
        loss = metrics_dict.get('loss', 0.0)
        lr = metrics_dict.get('lr', 0.0)
        image_auroc = metrics_dict.get('image_auroc', 0.0)
        pixel_auroc = metrics_dict.get('pixel_auroc', 0.0)
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{epoch},{phase},{loss:.6f},{lr:.8f},"
                   f"{image_auroc:.4f},{pixel_auroc:.4f}\n")
        
        # Store in memory
        self.metrics_history.append({
            'timestamp': timestamp,
            'epoch': epoch,
            'phase': phase,
            'metrics': metrics_dict.copy()
        })
    
    def get_best_metrics(self, metric_name='image_auroc', phase='val'):
        """Get best metrics for a specific metric and phase"""
        filtered_metrics = [
            m for m in self.metrics_history 
            if m['phase'] == phase and metric_name in m['metrics']
        ]
        
        if not filtered_metrics:
            return None
        
        best_entry = max(filtered_metrics, key=lambda x: x['metrics'][metric_name])
        return best_entry
    
    def get_latest_metrics(self, phase='val'):
        """Get latest metrics for a specific phase"""
        filtered_metrics = [
            m for m in self.metrics_history 
            if m['phase'] == phase
        ]
        
        if not filtered_metrics:
            return None
        
        return filtered_metrics[-1]


class ProgressTracker:
    """Track training progress"""
    
    def __init__(self, total_epochs, log_interval=10):
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        self.start_time = time.time()
        self.epoch_times = []
    
    def update(self, current_epoch, metrics=None):
        """
        Update progress tracker
        
        Args:
            current_epoch: Current epoch number
            metrics: Optional metrics dictionary
        """
        current_time = time.time()
        
        # Calculate epoch time
        if len(self.epoch_times) > 0:
            epoch_time = current_time - self.epoch_times[-1]
        else:
            epoch_time = current_time - self.start_time
        
        self.epoch_times.append(current_time)
        
        # Calculate estimates
        elapsed_time = current_time - self.start_time
        if current_epoch > 0:
            avg_epoch_time = elapsed_time / current_epoch
            remaining_epochs = self.total_epochs - current_epoch
            eta = remaining_epochs * avg_epoch_time
        else:
            eta = 0
        
        # Log progress
        if current_epoch % self.log_interval == 0 or current_epoch == self.total_epochs:
            progress_pct = (current_epoch / self.total_epochs) * 100
            
            elapsed_str = self._format_time(elapsed_time)
            eta_str = self._format_time(eta)
            
            logger = logging.getLogger("DMIAD_Progress")
            logger.info(f"Epoch {current_epoch}/{self.total_epochs} ({progress_pct:.1f}%) | "
                       f"Elapsed: {elapsed_str} | ETA: {eta_str}")
            
            if metrics:
                metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() 
                                       if isinstance(v, (int, float))])
                if metric_str:
                    logger.info(f"Metrics: {metric_str}")
    
    def _format_time(self, seconds):
        """Format time in human readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"


class ExperimentLogger:
    """Comprehensive experiment logging"""
    
    def __init__(self, exp_dir, config=None):
        self.exp_dir = exp_dir
        self.config = config
        
        # Create experiment directory
        os.makedirs(exp_dir, exist_ok=True)
        
        # Setup main logger
        self.logger = get_logger(exp_dir, 'experiment.log')
        
        # Setup metrics logger
        metrics_log_file = os.path.join(exp_dir, 'metrics.csv')
        self.metrics_logger = MetricsLogger(metrics_log_file)
        
        # Log experiment start
        self._log_experiment_start()
    
    def _log_experiment_start(self):
        """Log experiment start information"""
        self.logger.info("=" * 80)
        self.logger.info("DMIAD EXPERIMENT STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment Directory: {self.exp_dir}")
        self.logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.config:
            self.logger.info("Configuration:")
            self._log_config(self.config)
    
    def _log_config(self, config, prefix="  "):
        """Recursively log configuration"""
        if hasattr(config, '__dict__'):
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    self.logger.info(f"{prefix}{key}:")
                    self._log_config(value, prefix + "  ")
                else:
                    self.logger.info(f"{prefix}{key}: {value}")
        else:
            self.logger.info(f"{prefix}{config}")
    
    def log_epoch(self, epoch, phase, metrics_dict):
        """Log epoch metrics"""
        self.metrics_logger.log_metrics(epoch, phase, metrics_dict)
        
        # Log to main logger
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items() 
                               if isinstance(v, (int, float))])
        self.logger.info(f"Epoch {epoch} [{phase}] - {metric_str}")
    
    def log_best_result(self, metric_name='image_auroc'):
        """Log best result achieved"""
        best_metrics = self.metrics_logger.get_best_metrics(metric_name)
        if best_metrics:
            self.logger.info(f"Best {metric_name}: {best_metrics['metrics'][metric_name]:.4f} "
                           f"at epoch {best_metrics['epoch']}")
    
    def log_experiment_end(self):
        """Log experiment end"""
        self.logger.info("=" * 80)
        self.logger.info("DMIAD EXPERIMENT COMPLETED")
        self.logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Log best results
        self.log_best_result('image_auroc')
        self.log_best_result('pixel_auroc')
        
        self.logger.info("=" * 80)


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test basic logger
        logger = get_logger(temp_dir, 'test.log')
        logger.info("This is a test log message")
        logger.warning("This is a warning")
        logger.error("This is an error")
        
        # Test metrics logger
        metrics_logger = MetricsLogger(os.path.join(temp_dir, 'metrics.csv'))
        
        # Log some dummy metrics
        for epoch in range(5):
            train_metrics = {
                'loss': 1.0 - epoch * 0.1,
                'lr': 0.001,
            }
        # Log some dummy metrics
        for epoch in range(5):
            train_metrics = {
                'loss': 1.0 - epoch * 0.1,
                'lr': 0.001,
                'image_auroc': 0.5 + epoch * 0.1,
                'pixel_auroc': 0.6 + epoch * 0.08
            }
            
            val_metrics = {
                'loss': 0.8 - epoch * 0.05,
                'lr': 0.001,
                'image_auroc': 0.6 + epoch * 0.12,
                'pixel_auroc': 0.7 + epoch * 0.09
            }
            
            metrics_logger.log_metrics(epoch, 'train', train_metrics)
            metrics_logger.log_metrics(epoch, 'val', val_metrics)
        
        print("✓ Metrics logged successfully")
        
        # Test progress tracker
        progress_tracker = ProgressTracker(total_epochs=10, log_interval=2)
        
        for epoch in range(1, 6):
            time.sleep(0.1)  # Simulate training time
            progress_tracker.update(epoch, {'accuracy': 0.8 + epoch * 0.02})
        
        print("✓ Progress tracking test completed")
        
        # Test experiment logger
        config = type('Config', (), {
            'MODEL': type('Model', (), {'NAME': 'dmiad', 'BACKBONE': 'wide_resnet'})(),
            'DATASET': type('Dataset', (), {'NAME': 'mvtec', 'CLASS': 'bottle'})()
        })()
        
        exp_logger = ExperimentLogger(temp_dir, config)
        
        for epoch in range(3):
            metrics = {
                'loss': 1.0 - epoch * 0.2,
                'image_auroc': 0.7 + epoch * 0.1,
                'pixel_auroc': 0.75 + epoch * 0.08
            }
            exp_logger.log_epoch(epoch, 'val', metrics)
        
        exp_logger.log_experiment_end()
        print("✓ Experiment logging test completed")
    
    print("\nLogger utilities test completed!")