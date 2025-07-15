"""
Visualization utilities for DMIAD
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import cv2


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W]
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized tensor
    """
    if tensor.dim() == 4:
        # Batch of images
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    else:
        # Single image
        mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    
    return tensor * std + mean


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization"""
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Handle different tensor shapes
    if tensor.dim() == 4:  # [B, C, H, W]
        tensor = tensor[0]  # Take first image in batch
    
    if tensor.dim() == 3:  # [C, H, W]
        if tensor.shape[0] == 3:  # RGB image
            tensor = tensor.permute(1, 2, 0)  # [H, W, C]
        elif tensor.shape[0] == 1:  # Grayscale
            tensor = tensor.squeeze(0)  # [H, W]
    
    numpy_array = tensor.numpy()
    
    # Clip values to valid range
    if tensor.dim() == 3 and tensor.shape[-1] == 3:  # RGB
        numpy_array = np.clip(numpy_array, 0, 1)
    
    return numpy_array


def save_image_comparison(original, reconstructed, anomaly_map, save_path, 
                         ground_truth=None, title=None):
    """
    Save comparison of original, reconstructed, and anomaly map
    
    Args:
        original: Original image tensor
        reconstructed: Reconstructed image tensor
        anomaly_map: Anomaly score map tensor
        save_path: Path to save image
        ground_truth: Ground truth mask (optional)
        title: Plot title (optional)
    """
    # Convert tensors to numpy
    orig_np = tensor_to_numpy(denormalize_image(original))
    recon_np = tensor_to_numpy(denormalize_image(reconstructed))
    anomaly_np = tensor_to_numpy(anomaly_map)
    
    # Setup subplot grid
    num_cols = 4 if ground_truth is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    
    if num_cols == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(orig_np)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Reconstructed image
    axes[1].imshow(recon_np)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    # Anomaly map
    im2 = axes[2].imshow(anomaly_np, cmap='jet', alpha=0.8)
    axes[2].set_title('Anomaly Map')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Ground truth (if provided)
    if ground_truth is not None:
        gt_np = tensor_to_numpy(ground_truth)
        axes[3].imshow(gt_np, cmap='gray')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_attention_visualization(attention_maps, save_path, titles=None):
    """
    Visualize attention maps from memory modules
    
    Args:
        attention_maps: Dict or list of attention tensors
        save_path: Path to save visualization
        titles: List of titles for each attention map
    """
    if isinstance(attention_maps, dict):
        att_list = list(attention_maps.values())
        if titles is None:
            titles = list(attention_maps.keys())
    else:
        att_list = attention_maps
        if titles is None:
            titles = [f'Attention {i}' for i in range(len(att_list))]
    
    num_maps = len(att_list)
    if num_maps == 0:
        return
    
    # Create subplot grid
    cols = min(4, num_maps)
    rows = (num_maps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (att_map, title) in enumerate(zip(att_list, titles)):
        if i >= len(axes):
            break
        
        # Convert attention map to numpy
        att_np = tensor_to_numpy(att_map)
        
        # If attention map has multiple channels, take mean
        if att_np.ndim == 3:
            att_np = np.mean(att_np, axis=0)
        elif att_np.ndim == 3 and att_np.shape[0] > 1:
            att_np = np.mean(att_np, axis=0)
        
        # Normalize attention map
        att_np = (att_np - att_np.min()) / (att_np.max() - att_np.min() + 1e-8)
        
        # Visualize
        im = axes[i].imshow(att_np, cmap='hot', interpolation='nearest')
        axes[i].set_title(title)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(len(att_list), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_memory_analysis(memory_items, save_path):
    """
    Visualize memory bank patterns
    
    Args:
        memory_items: Dict of memory tensors
        save_path: Path to save visualization
    """
    num_memories = len(memory_items)
    if num_memories == 0:
        return
    
    fig, axes = plt.subplots(2, num_memories, figsize=(6 * num_memories, 8))
    
    if num_memories == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (mem_name, mem_tensor) in enumerate(memory_items.items()):
        mem_np = tensor_to_numpy(mem_tensor)
        
        # Memory patterns visualization
        if mem_np.ndim == 2:  # [memory_size, feature_dim]
            # Show memory bank as heatmap
            im1 = axes[0, i].imshow(mem_np, cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'{mem_name} Patterns')
            axes[0, i].set_xlabel('Feature Dimension')
            axes[0, i].set_ylabel('Memory Index')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Memory similarity matrix
            mem_norm = mem_np / (np.linalg.norm(mem_np, axis=1, keepdims=True) + 1e-8)
            similarity = np.dot(mem_norm, mem_norm.T)
            
            im2 = axes[1, i].imshow(similarity, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, i].set_title(f'{mem_name} Similarity')
            axes[1, i].set_xlabel('Memory Index')
            axes[1, i].set_ylabel('Memory Index')
            plt.colorbar(im2, ax=axes[1, i])
        
        else:
            # For other shapes, show flattened version
            mem_flat = mem_np.reshape(mem_np.shape[0], -1)
            im1 = axes[0, i].imshow(mem_flat, cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'{mem_name} (Flattened)')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Show mean activation
            mean_activation = np.mean(mem_flat, axis=0)
            axes[1, i].plot(mean_activation)
            axes[1, i].set_title(f'{mem_name} Mean Activation')
            axes[1, i].set_xlabel('Feature Index')
            axes[1, i].set_ylabel('Activation')
            axes[1, i].grid(True)
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_anomaly_heatmap(image, anomaly_scores, alpha=0.4, colormap='jet'):
    """
    Create heatmap overlay of anomaly scores on original image
    
    Args:
        image: Original image [H, W, 3] or [3, H, W]
        anomaly_scores: Anomaly score map [H, W] or [1, H, W]
        alpha: Transparency of heatmap overlay
        colormap: Matplotlib colormap name
    
    Returns:
        numpy array: Image with heatmap overlay
    """
    # Convert to numpy and ensure correct format
    if isinstance(image, torch.Tensor):
        image = tensor_to_numpy(denormalize_image(image))
    
    if isinstance(anomaly_scores, torch.Tensor):
        anomaly_scores = tensor_to_numpy(anomaly_scores)
    
    # Ensure image is [H, W, 3]
    if image.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
        image = np.transpose(image, (1, 2, 0))
    
    # Ensure anomaly_scores is [H, W]
    if anomaly_scores.ndim == 3:
        anomaly_scores = anomaly_scores.squeeze()
    
    # Normalize anomaly scores
    norm_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
    
    # Create colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(norm_scores)[:, :, :3]  # Remove alpha channel
    
    # Blend with original image
    blended = (1 - alpha) * image + alpha * heatmap
    
    return np.clip(blended, 0, 1)


def save_test_visualizations(results, save_dir, num_samples=10):
    """
    Save comprehensive test visualizations
    
    Args:
        results: Test results dictionary
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Select samples to visualize
    total_samples = len(results['image_paths'])
    indices = np.linspace(0, total_samples - 1, min(num_samples, total_samples), dtype=int)
    
    for idx in indices:
        image_path = results['image_paths'][idx]
        image_name = os.path.basename(image_path).split('.')[0]
        
        # Load original image
        original_img = Image.open(image_path).convert('RGB')
        original_tensor = TF.to_tensor(original_img).unsqueeze(0)
        
        # Get results for this sample
        if 'reconstructions' in results:
            reconstructed = results['reconstructions'][idx].unsqueeze(0)
        else:
            reconstructed = original_tensor  # Fallback
        
        # Create anomaly map (placeholder if not available)
        anomaly_score = results['image_scores'][idx]
        H, W = original_tensor.shape[-2:]
        anomaly_map = torch.full((1, 1, H, W), anomaly_score)
        
        # Ground truth mask (if available)
        ground_truth = None
        if 'pixel_labels' in results and idx < len(results['pixel_labels']):
            # Reconstruct ground truth mask
            # This is a simplified version - you might need to adjust based on your data structure
            gt_data = results['pixel_labels'][idx] if isinstance(results['pixel_labels'][idx], np.ndarray) else None
            if gt_data is not None:
                ground_truth = torch.tensor(gt_data).unsqueeze(0).unsqueeze(0)
        
        # Save comparison
        save_path = os.path.join(save_dir, f'{image_name}_comparison.png')
        save_image_comparison(
            original_tensor[0], reconstructed[0], anomaly_map[0], 
            save_path, ground_truth[0] if ground_truth is not None else None,
            title=f'{image_name} (Score: {anomaly_score:.3f})'
        )
        
        # Save attention maps if available
        if 'attention_maps' in results and idx < len(results['attention_maps']):
            attention_data = results['attention_maps'][idx]
            if attention_data:
                att_save_path = os.path.join(save_dir, f'{image_name}_attention.png')
                save_attention_visualization(attention_data, att_save_path)


def plot_score_distribution(normal_scores, anomaly_scores, save_path=None, title='Score Distribution'):
    """
    Plot distribution of anomaly scores for normal and anomaly samples
    
    Args:
        normal_scores: List/array of scores for normal samples
        anomaly_scores: List/array of scores for anomaly samples
        save_path: Path to save plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Convert to numpy arrays and check if not empty
    if normal_scores is not None and len(normal_scores) > 0:
        normal_scores = np.array(normal_scores)
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    
    if anomaly_scores is not None and len(anomaly_scores) > 0:
        anomaly_scores = np.array(anomaly_scores)
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy data for testing
        original = torch.randn(3, 256, 256)
        reconstructed = original + torch.randn_like(original) * 0.1
        anomaly_map = torch.randn(1, 256, 256).abs()
        ground_truth = torch.randint(0, 2, (1, 256, 256)).float()
        
        # Test image comparison
        save_path = os.path.join(temp_dir, 'comparison.png')
        save_image_comparison(original, reconstructed, anomaly_map, save_path, ground_truth)
        print(f"✓ Image comparison saved to {save_path}")
        
        # Test attention visualization
        attention_maps = {
            'temporal': torch.randn(1, 64, 64),
            'spatial': torch.randn(1, 64, 64)
        }
        att_save_path = os.path.join(temp_dir, 'attention.png')
        save_attention_visualization(attention_maps, att_save_path)
        print(f"✓ Attention visualization saved to {att_save_path}")
        
        # Test memory analysis
        memory_items = {
            'temporal_memory': torch.randn(2000, 512),
            'spatial_memory': torch.randn(2000, 512)
        }
        mem_save_path = os.path.join(temp_dir, 'memory_analysis.png')
        save_memory_analysis(memory_items, mem_save_path)
        print(f"✓ Memory analysis saved to {mem_save_path}")
        
        # Test score distribution
        normal_scores = np.random.normal(0.3, 0.1, 100)
        anomaly_scores = np.random.normal(0.7, 0.15, 50)
        dist_save_path = os.path.join(temp_dir, 'score_distribution.png')
        plot_score_distribution(normal_scores, anomaly_scores, dist_save_path)
        print(f"✓ Score distribution saved to {dist_save_path}")
    
    print("\nVisualization utilities test completed!")