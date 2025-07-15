"""
Testing script for DMIAD
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.mvtec import get_dataloaders, DatasetSplit
from models.dmiad_model import build_dmiad_model
from utils.metrics import compute_comprehensive_metrics
from utils.checkpoint import load_checkpoint
from utils.logger import get_logger
from utils.visualization import save_test_visualizations


class Tester:
    """Testing class for DMIAD"""
    
    def __init__(self, config, device, exp_dir):
        self.config = config
        self.device = device
        self.exp_dir = exp_dir
        
        # Setup logging
        self.logger = get_logger(exp_dir, filename='test.log')
        
        # Create model
        self.model = build_dmiad_model(config).to(device)
        
        # Load trained model
        self._load_model()
        
        # Create test dataloader
        self.test_loader = get_dataloaders(config)['test']
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
        
        # Results storage
        self.results = {
            'image_scores': [],
            'image_labels': [],
            'pixel_scores': [],
            'pixel_labels': [],
            'image_paths': [],
            'reconstructions': [],
            'attention_maps': [],
        }
    
    def _load_model(self):
        """Load trained model"""
        if self.config.TEST.CHECKPOINT:
            checkpoint_path = self.config.TEST.CHECKPOINT
        else:
            # Try to find best checkpoint
            checkpoint_path = os.path.join(self.exp_dir, 'best_checkpoint.pth')
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(self.exp_dir, 'latest_checkpoint.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        self.logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Log training info if available
        if 'metrics' in checkpoint:
            self.logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
            self.logger.info(f"Training metrics: {checkpoint['metrics']}")
    
    def run_inference(self):
        """Run inference on test set"""
        self.logger.info("Starting inference...")
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Testing')):
                # Move data to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)
                image_paths = batch['image_path']
                
                # Forward pass
                outputs = self.model(images)
                reconstructed = outputs['reconstructed']
                memory_results = outputs['memory_results']
                
                # Compute anomaly scores
                anomaly_scores = self.model.compute_anomaly_score(images, reconstructed)
                
                # Store results
                self._store_batch_results(
                    images, reconstructed, masks, labels,
                    anomaly_scores, memory_results, image_paths
                )
        
        self.logger.info("Inference completed!")
    
    def _store_batch_results(self, images, reconstructed, masks, labels, 
                           anomaly_scores, memory_results, image_paths):
        """Store results from a batch"""
        
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            # Image-level results
            image_score = anomaly_scores['image_scores'][i].cpu().item()
            image_label = labels[i].cpu().item()
            
            self.results['image_scores'].append(image_score)
            self.results['image_labels'].append(image_label)
            self.results['image_paths'].append(image_paths[i])
            
            # Pixel-level results
            pixel_score_map = anomaly_scores['pixel_scores'][i].cpu().numpy()
            pixel_label_map = masks[i].cpu().numpy()
            
            # Flatten and store pixel scores/labels
            pixel_scores_flat = pixel_score_map.flatten()
            pixel_labels_flat = pixel_label_map.flatten()
            
            self.results['pixel_scores'].extend(pixel_scores_flat)
            self.results['pixel_labels'].extend(pixel_labels_flat)
            
            # Store reconstructions and attention maps for visualization
            if self.config.TEST.SAVE_IMAGES:
                reconstruction = reconstructed[i].cpu()
                self.results['reconstructions'].append(reconstruction)
                
                # Store attention maps
                attention_data = {}
                if 'temporal_attention' in memory_results:
                    attention_data['temporal'] = memory_results['temporal_attention'][i].cpu()
                if 'spatial_attention' in memory_results:
                    attention_data['spatial'] = memory_results['spatial_attention'][i].cpu()
                
                self.results['attention_maps'].append(attention_data)
    
    def compute_metrics(self):
        """Compute comprehensive evaluation metrics"""
        self.logger.info("Computing metrics...")
        
        # Convert to numpy arrays
        image_scores = np.array(self.results['image_scores'])
        image_labels = np.array(self.results['image_labels'])
        pixel_scores = np.array(self.results['pixel_scores'])
        pixel_labels = np.array(self.results['pixel_labels'])
        
        # Compute metrics
        metrics = compute_comprehensive_metrics(
            image_scores, image_labels,
            pixel_scores, pixel_labels
        )
        
        # Log metrics
        self.logger.info("=== Test Results ===")
        self.logger.info(f"Image-level AUROC: {metrics['image_auroc']:.4f}")
        self.logger.info(f"Image-level AUPR: {metrics['image_aupr']:.4f}")
        self.logger.info(f"Pixel-level AUROC: {metrics['pixel_auroc']:.4f}")
        self.logger.info(f"Pixel-level AUPR: {metrics['pixel_aupr']:.4f}")
        
        if 'pixel_pro' in metrics:
            self.logger.info(f"Pixel-level PRO: {metrics['pixel_pro']:.4f}")
        
        # Save metrics to file
        metrics_file = os.path.join(self.exp_dir, 'test_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
        
        return metrics
    
    def save_visualizations(self):
        """Save test visualizations"""
        if not self.config.TEST.SAVE_IMAGES:
            return
        
        self.logger.info("Saving visualizations...")
        
        vis_dir = os.path.join(self.exp_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save sample visualizations
        self._save_sample_visualizations(vis_dir)
        
        # Save attention visualizations
        self._save_attention_visualizations(vis_dir)
        
        # Save score distribution plots
        self._save_score_distributions(vis_dir)
        
        self.logger.info(f"Visualizations saved to {vis_dir}")
    
    def _save_sample_visualizations(self, vis_dir):
        """Save sample reconstructions and anomaly maps"""
        
        sample_dir = os.path.join(vis_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Select samples to visualize
        num_samples = min(20, len(self.results['image_paths']))
        indices = np.linspace(0, len(self.results['image_paths'])-1, num_samples, dtype=int)
        
        for idx in indices:
            image_path = self.results['image_paths'][idx]
            image_name = os.path.basename(image_path).split('.')[0]
            
            # TODO: Implement sample visualization
            # - Original image
            # - Reconstruction
            # - Anomaly score map
            # - Difference map
            save_path = os.path.join(sample_dir, f'{image_name}_reconstruction.png')
            # save_test_visualizations(original, reconstruction, anomaly_map, save_path)
    
    def _save_attention_visualizations(self, vis_dir):
        """Save attention map visualizations"""
        
        attention_dir = os.path.join(vis_dir, 'attention')
        os.makedirs(attention_dir, exist_ok=True)
        
        # Select samples with attention maps
        num_samples = min(10, len(self.results['attention_maps']))
        
        for idx in range(num_samples):
            attention_data = self.results['attention_maps'][idx]
            image_path = self.results['image_paths'][idx]
            image_name = os.path.basename(image_path).split('.')[0]
            
            # Save temporal attention maps
            if 'temporal' in attention_data:
                # TODO: Visualize temporal attention maps
                pass
            
            # Save spatial attention maps  
            if 'spatial' in attention_data:
                # TODO: Visualize spatial attention maps
                pass
    
    def _save_score_distributions(self, vis_dir):
        """Save anomaly score distribution plots"""
        
        # Separate normal and anomaly scores
        normal_scores = []
        anomaly_scores = []
        
        for score, label in zip(self.results['image_scores'], self.results['image_labels']):
            if label == 0:
                normal_scores.append(score)
            else:
                anomaly_scores.append(score)
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        
        if normal_scores:
            plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        if anomaly_scores:
            plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_memory_patterns(self):
        """Analyze memory patterns"""
        self.logger.info("Analyzing memory patterns...")
        
        memory_items = self.model.get_memory_items()
        analysis_results = {}
        
        for mem_name, mem_tensor in memory_items.items():
            # Compute memory statistics
            mem_stats = {
                'shape': list(mem_tensor.shape),
                'mean': float(torch.mean(mem_tensor)),
                'std': float(torch.std(mem_tensor)),
                'min': float(torch.min(mem_tensor)),
                'max': float(torch.max(mem_tensor)),
                'norm': float(torch.norm(mem_tensor)),
            }
            
            # Compute memory item similarities
            mem_normalized = torch.nn.functional.normalize(mem_tensor, dim=1)
            similarities = torch.mm(mem_normalized, mem_normalized.t())
            
            # Exclude diagonal (self-similarities)
            mask = ~torch.eye(similarities.shape[0], dtype=torch.bool)
            off_diagonal_similarities = similarities[mask]
            
            mem_stats['similarity_mean'] = float(torch.mean(off_diagonal_similarities))
            mem_stats['similarity_std'] = float(torch.std(off_diagonal_similarities))
            
            analysis_results[mem_name] = mem_stats
            
            self.logger.info(f"{mem_name} - Shape: {mem_stats['shape']}, "
                           f"Norm: {mem_stats['norm']:.4f}, "
                           f"Similarity Mean: {mem_stats['similarity_mean']:.4f}")
        
        # Save memory analysis
        memory_file = os.path.join(self.exp_dir, 'memory_analysis.json')
        with open(memory_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        return analysis_results
    
    def generate_report(self, metrics):
        """Generate comprehensive test report"""
        self.logger.info("Generating test report...")
        
        report = {
            'experiment_info': {
                'dataset': self.config.DATASET.NAME,
                'class_name': self.config.DATASET.CLASS_NAME,
                'setting': self.config.DATASET.SETTING,
                'model_name': self.config.MODEL.NAME,
                'use_spatial_memory': self.config.MODEL.MEMORY.USE_SPATIAL,
                'fusion_method': self.config.MODEL.MEMORY.FUSION_METHOD,
            },
            'test_metrics': metrics,
            'dataset_stats': {
                'total_samples': len(self.results['image_labels']),
                'normal_samples': sum(1 for x in self.results['image_labels'] if x == 0),
                'anomaly_samples': sum(1 for x in self.results['image_labels'] if x == 1),
            }
        }
        
        # Add memory analysis if available
        if hasattr(self, 'memory_analysis'):
            report['memory_analysis'] = self.memory_analysis
        
        # Save report
        report_file = os.path.join(self.exp_dir, 'test_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Test report saved to {report_file}")
        
        return report
    
    def test(self):
        """Main testing pipeline"""
        self.logger.info("Starting testing pipeline...")
        
        # 1. Run inference
        self.run_inference()
        
        # 2. Compute metrics
        metrics = self.compute_metrics()
        
        # 3. Save visualizations
        self.save_visualizations()
        
        # 4. Analyze memory patterns
        self.memory_analysis = self.analyze_memory_patterns()
        
        # 5. Generate comprehensive report
        report = self.generate_report(metrics)
        
        self.logger.info("Testing pipeline completed!")
        
        return metrics, report


def test_model(config, device, exp_dir):
    """Main testing function"""
    
    # Create tester
    tester = Tester(config, device, exp_dir)
    
    # Run testing
    metrics, report = tester.test()
    
    return tester, metrics, report


class BatchTester:
    """Test multiple models/settings in batch"""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = {}
    
    def test_multiple_classes(self, class_names, device):
        """Test on multiple classes"""
        
        for class_name in class_names:
            print(f"\n=== Testing on {class_name} ===")
            
            # Update config for current class
            config = self.base_config.copy()
            config.DATASET.CLASS_NAME = class_name
            
            # Create experiment directory
            exp_dir = os.path.join(
                config.OUTPUT_DIR, 
                f"{config.EXP_NAME}_{class_name}"
            )
            
            try:
                # Run testing
                tester, metrics, report = test_model(config, device, exp_dir)
                
                # Store results
                self.results[class_name] = {
                    'metrics': metrics,
                    'report': report
                }
                
                print(f"✓ {class_name}: Image AUROC = {metrics['image_auroc']:.4f}, "
                      f"Pixel AUROC = {metrics['pixel_auroc']:.4f}")
                
            except Exception as e:
                print(f"✗ {class_name}: Failed with error: {str(e)}")
                self.results[class_name] = {'error': str(e)}
        
        # Save combined results
        self._save_combined_results()
    
    def _save_combined_results(self):
        """Save combined results across all classes"""
        
        # Compute average metrics
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if valid_results:
            avg_metrics = {}
            metric_names = list(valid_results.values())[0]['metrics'].keys()
            
            for metric_name in metric_names:
                values = [result['metrics'][metric_name] for result in valid_results.values()]
                avg_metrics[metric_name] = np.mean(values)
            
            # Create summary
            summary = {
                'average_metrics': avg_metrics,
                'per_class_results': self.results,
                'successful_classes': len(valid_results),
                'failed_classes': len(self.results) - len(valid_results)
            }
            
            # Save summary
            summary_file = os.path.join(self.base_config.OUTPUT_DIR, 'batch_test_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n=== Batch Testing Summary ===")
            print(f"Average Image AUROC: {avg_metrics['image_auroc']:.4f}")
            print(f"Average Pixel AUROC: {avg_metrics['pixel_auroc']:.4f}")
            print(f"Summary saved to: {summary_file}")


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
    args.batch_size = 1
    args.gpu = 0
    args.mem_dim = 2000
    args.use_spatial_memory = True
    args.fusion_method = 'add'
    args.backbone = 'wide_resnet'
    args.epochs = 100
    args.lr = 1e-4
    args.output_dir = './results'
    args.exp_name = 'test_dmiad'
    args.save_images = True
    args.checkpoint = None
    
    # Get config
    config = get_config(args)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment directory
    exp_dir = os.path.join(config.OUTPUT_DIR, config.EXP_NAME)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Run testing
    tester, metrics, report = test_model(config, device, exp_dir)
    
    print("\n=== Test Results ===")
    print(f"Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"Pixel AUROC: {metrics['pixel_auroc']:.4f}")