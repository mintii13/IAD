"""
Evaluation Metrics for DMIAD
"""

from sklearn import metrics
from skimage import measure
import cv2
import numpy as np
import pandas as pd


def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, thresholds = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_precision = precision[np.argmax(f1_scores)]
    best_recall = recall[np.argmax(f1_scores)]

    return best_threshold, best_precision, best_recall


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).
    """
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    return {"auroc": auroc, "ap": ap}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train'):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = metrics.roc_auc_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)
    ap = 0. if path == 'training' else metrics.average_precision_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    return {"auroc": auroc, "ap": ap}


def compute_pro(masks, amaps, num_th=200):
    """
    Compute Per-Region Overlap (PRO) metric
    """
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / (inverse_masks.sum() + 1e-10)

        df = df.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    if len(df) > 0:
        df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)
        pro_auc = metrics.auc(df["fpr"], df["pro"])
    else:
        pro_auc = 0.0
    
    return pro_auc


def compute_metrics(image_scores, image_labels, pixel_scores=None, pixel_labels=None):
    """
    Compute comprehensive metrics for anomaly detection
    
    Args:
        image_scores: Image-level anomaly scores
        image_labels: Image-level ground truth labels
        pixel_scores: Pixel-level anomaly scores (optional)
        pixel_labels: Pixel-level ground truth labels (optional)
    
    Returns:
        Dict of computed metrics
    """
    metrics_dict = {}
    
    # Image-level metrics
    if len(np.unique(image_labels)) > 1:  # Check if both classes exist
        image_metrics = compute_imagewise_retrieval_metrics(
            image_scores, image_labels, path='test'
        )
        metrics_dict['image_auroc'] = image_metrics['auroc']
        metrics_dict['image_aupr'] = image_metrics['ap']
        
        # Best threshold metrics
        best_th, best_pr, best_re = compute_best_pr_re(image_labels, image_scores)
        metrics_dict['best_threshold'] = best_th
        metrics_dict['best_precision'] = best_pr
        metrics_dict['best_recall'] = best_re
        metrics_dict['best_f1'] = 2 * best_pr * best_re / (best_pr + best_re + 1e-10)
    else:
        # If only one class, set default values
        metrics_dict['image_auroc'] = 0.5
        metrics_dict['image_aupr'] = 0.0
        metrics_dict['best_threshold'] = 0.5
        metrics_dict['best_precision'] = 0.0
        metrics_dict['best_recall'] = 0.0
        metrics_dict['best_f1'] = 0.0
    
    # Pixel-level metrics
    if pixel_scores is not None and pixel_labels is not None:
        if len(np.unique(pixel_labels)) > 1:  # Check if both classes exist
            pixel_metrics = compute_pixelwise_retrieval_metrics(
                pixel_scores, pixel_labels, path='test'
            )
            metrics_dict['pixel_auroc'] = pixel_metrics['auroc']
            metrics_dict['pixel_aupr'] = pixel_metrics['ap']
        else:
            metrics_dict['pixel_auroc'] = 0.5
            metrics_dict['pixel_aupr'] = 0.0
    
    return metrics_dict


def compute_comprehensive_metrics(image_scores, image_labels, pixel_scores, pixel_labels):
    """
    Compute comprehensive evaluation metrics including PRO
    
    Args:
        image_scores: [N] - Image-level anomaly scores
        image_labels: [N] - Image-level ground truth labels
        pixel_scores: [N*H*W] - Pixel-level anomaly scores (flattened)
        pixel_labels: [N*H*W] - Pixel-level ground truth labels (flattened)
    
    Returns:
        Dict of comprehensive metrics
    """
    metrics_dict = compute_metrics(image_scores, image_labels, pixel_scores, pixel_labels)
    
    # Add PRO metric if pixel-level data available
    if pixel_scores is not None and pixel_labels is not None:
        try:
            # Reshape pixel data back to image format for PRO calculation
            # This assumes square images, you might need to adjust based on your data
            num_images = len(image_scores)
            pixels_per_image = len(pixel_scores) // num_images
            img_size = int(np.sqrt(pixels_per_image))
            
            if img_size * img_size == pixels_per_image:
                pixel_scores_reshaped = pixel_scores.reshape(num_images, img_size, img_size)
                pixel_labels_reshaped = pixel_labels.reshape(num_images, img_size, img_size)
                
                # Only compute PRO for images with anomalies
                anomaly_indices = np.where(image_labels == 1)[0]
                if len(anomaly_indices) > 0:
                    anomaly_pixel_scores = pixel_scores_reshaped[anomaly_indices]
                    anomaly_pixel_labels = pixel_labels_reshaped[anomaly_indices]
                    
                    pro_auc = compute_pro(anomaly_pixel_labels, anomaly_pixel_scores)
                    metrics_dict['pixel_pro'] = pro_auc
                else:
                    metrics_dict['pixel_pro'] = 0.0
            else:
                metrics_dict['pixel_pro'] = 0.0
        except Exception as e:
            print(f"Warning: Could not compute PRO metric: {e}")
            metrics_dict['pixel_pro'] = 0.0
    
    return metrics_dict


def print_metrics(metrics_dict, title="Evaluation Metrics"):
    """Pretty print metrics"""
    print(f"\n=== {title} ===")
    print("-" * 50)
    
    # Image-level metrics
    if 'image_auroc' in metrics_dict:
        print(f"Image AUROC:     {metrics_dict['image_auroc']:.4f}")
    if 'image_aupr' in metrics_dict:
        print(f"Image AUPR:      {metrics_dict['image_aupr']:.4f}")
    
    # Pixel-level metrics
    if 'pixel_auroc' in metrics_dict:
        print(f"Pixel AUROC:     {metrics_dict['pixel_auroc']:.4f}")
    if 'pixel_aupr' in metrics_dict:
        print(f"Pixel AUPR:      {metrics_dict['pixel_aupr']:.4f}")
    if 'pixel_pro' in metrics_dict:
        print(f"Pixel PRO:       {metrics_dict['pixel_pro']:.4f}")
    
    # Best threshold metrics
    if 'best_f1' in metrics_dict:
        print(f"Best F1:         {metrics_dict['best_f1']:.4f}")
    if 'best_threshold' in metrics_dict:
        print(f"Best Threshold:  {metrics_dict['best_threshold']:.4f}")
    
    print("-" * 50)


# Example usage and testing
if __name__ == "__main__":
    # Generate dummy data for testing
    np.random.seed(42)
    
    # Image-level data
    n_images = 100
    image_labels = np.random.choice([0, 1], size=n_images, p=[0.7, 0.3])
    image_scores = np.random.random(n_images)
    # Make anomaly scores generally higher
    image_scores[image_labels == 1] += 0.3
    image_scores = np.clip(image_scores, 0, 1)
    
    # Pixel-level data (assuming 64x64 images)
    img_size = 64
    n_pixels = n_images * img_size * img_size
    pixel_labels = np.random.choice([0, 1], size=n_pixels, p=[0.95, 0.05])
    pixel_scores = np.random.random(n_pixels)
    # Make anomaly pixels generally higher scores
    pixel_scores[pixel_labels == 1] += 0.4
    pixel_scores = np.clip(pixel_scores, 0, 1)
    
    # Test basic metrics
    basic_metrics = compute_metrics(image_scores, image_labels, pixel_scores, pixel_labels)
    print_metrics(basic_metrics, "Basic Metrics Test")
    
    # Test comprehensive metrics
    comprehensive_metrics = compute_comprehensive_metrics(
        image_scores, image_labels, pixel_scores, pixel_labels
    )
    print_metrics(comprehensive_metrics, "Comprehensive Metrics Test")
    
    print("\nMetrics module test completed!")