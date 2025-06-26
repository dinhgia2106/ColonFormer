"""
Evaluation metrics for ColonFormer
Implementation of common segmentation metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def calculate_metrics(predictions, targets, threshold=0.5, smooth=1e-7):
    """
    Calculate comprehensive metrics for binary segmentation
    
    Args:
        predictions: predicted probabilities [B, 1, H, W] (0-1)
        targets: ground truth masks [B, 1, H, W] (0-1)
        threshold: threshold for binary prediction
        smooth: smoothing factor
    
    Returns:
        dict: metrics dictionary
    """
    # Convert to binary predictions
    preds_binary = (predictions > threshold).float()
    
    # Flatten tensors for calculation
    preds_flat = predictions.view(-1)
    targets_flat = targets.view(-1)
    preds_binary_flat = preds_binary.view(-1)
    
    # True Positives, False Positives, False Negatives
    tp = (preds_binary_flat * targets_flat).sum()
    fp = (preds_binary_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_binary_flat) * targets_flat).sum()
    tn = ((1 - preds_binary_flat) * (1 - targets_flat)).sum()
    
    # Dice Score (F1-Score)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    # IoU (Jaccard Index)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    
    # Precision
    precision = (tp + smooth) / (tp + fp + smooth)
    
    # Recall (Sensitivity)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    # Specificity
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    # Accuracy
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    
    # F2 Score (weighted towards recall)
    f2 = (5 * tp + smooth) / (5 * tp + 4 * fn + fp + smooth)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'specificity': specificity.item(),
        'accuracy': accuracy.item(),
        'f2': f2.item()
    }


def calculate_hausdorff_distance(pred, target, spacing=None):
    """
    Calculate Hausdorff distance between prediction and target
    
    Args:
        pred: binary prediction [H, W]
        target: binary target [H, W]
        spacing: pixel spacing, default (1, 1)
    
    Returns:
        float: Hausdorff distance
    """
    try:
        from scipy.spatial.distance import directed_hausdorff
        
        if spacing is None:
            spacing = [1, 1]
        
        # Convert to numpy and get coordinates of boundary points
        pred_np = pred.cpu().numpy().astype(bool)
        target_np = target.cpu().numpy().astype(bool)
        
        # Get boundary points
        pred_points = np.argwhere(pred_np) * spacing
        target_points = np.argwhere(target_np) * spacing
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(hd1, hd2)
    
    except ImportError:
        # Fallback if scipy is not available
        return 0.0


def calculate_mae(predictions, targets):
    """
    Calculate Mean Absolute Error
    
    Args:
        predictions: predicted probabilities [B, 1, H, W]
        targets: ground truth masks [B, 1, H, W]
    
    Returns:
        float: MAE value
    """
    mae = torch.mean(torch.abs(predictions - targets))
    return mae.item()


def calculate_enhanced_metrics(predictions, targets, threshold=0.5):
    """
    Calculate enhanced metrics including boundary-aware metrics
    
    Args:
        predictions: predicted probabilities [B, 1, H, W]
        targets: ground truth masks [B, 1, H, W]
        threshold: threshold for binary prediction
    
    Returns:
        dict: enhanced metrics dictionary
    """
    basic_metrics = calculate_metrics(predictions, targets, threshold)
    
    # Calculate MAE
    mae = calculate_mae(predictions, targets)
    
    # Boundary IoU (IoU calculated only on boundary regions)
    boundary_iou = calculate_boundary_iou(predictions, targets, threshold)
    
    # Enhanced metrics
    enhanced_metrics = basic_metrics.copy()
    enhanced_metrics.update({
        'mae': mae,
        'boundary_iou': boundary_iou
    })
    
    return enhanced_metrics


def calculate_boundary_iou(predictions, targets, threshold=0.5, kernel_size=3):
    """
    Calculate IoU specifically for boundary regions
    
    Args:
        predictions: predicted probabilities [B, 1, H, W]
        targets: ground truth masks [B, 1, H, W]
        threshold: threshold for binary prediction
        kernel_size: kernel size for boundary extraction
    
    Returns:
        float: boundary IoU
    """
    # Convert to binary
    preds_binary = (predictions > threshold).float()
    
    # Extract boundaries using morphological operations
    def get_boundary(mask):
        # Erosion
        eroded = F.max_pool2d(-mask, kernel_size, stride=1, padding=kernel_size//2)
        eroded = -eroded
        # Boundary = original - eroded
        boundary = mask - eroded
        return (boundary > 0).float()
    
    pred_boundary = get_boundary(preds_binary)
    target_boundary = get_boundary(targets)
    
    # Calculate IoU on boundaries
    intersection = (pred_boundary * target_boundary).sum()
    union = (pred_boundary + target_boundary - pred_boundary * target_boundary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    boundary_iou = intersection / union
    return boundary_iou.item()


class MetricsTracker:
    """Track metrics across multiple batches"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.metrics = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'accuracy': [],
            'f2': [],
            'mae': [],
            'boundary_iou': []
        }
    
    def update(self, predictions, targets, threshold=0.5):
        """Update metrics with new batch"""
        batch_metrics = calculate_enhanced_metrics(predictions, targets, threshold)
        
        for key, value in batch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_average_metrics(self):
        """Get average metrics across all batches"""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
            else:
                avg_metrics[key] = 0.0
                avg_metrics[f'{key}_std'] = 0.0
        
        return avg_metrics
    
    def get_best_metrics(self):
        """Get best metrics across all batches"""
        best_metrics = {}
        for key, values in self.metrics.items():
            if values:
                best_metrics[f'best_{key}'] = np.max(values)
            else:
                best_metrics[f'best_{key}'] = 0.0
        
        return best_metrics


def calculate_pr_auc(predictions, targets):
    """
    Calculate Precision-Recall AUC
    
    Args:
        predictions: predicted probabilities [B, 1, H, W]
        targets: ground truth masks [B, 1, H, W]
    
    Returns:
        float: PR-AUC value
    """
    # Flatten
    preds_flat = predictions.view(-1).cpu().numpy()
    targets_flat = targets.view(-1).cpu().numpy()
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(targets_flat, preds_flat)
    
    # Calculate AUC
    pr_auc = auc(recall, precision)
    
    return pr_auc


def calculate_dataset_metrics(model, dataloader, device, threshold=0.5):
    """
    Calculate metrics across entire dataset
    
    Args:
        model: trained model
        dataloader: data loader
        device: computation device
        threshold: binary threshold
    
    Returns:
        dict: comprehensive metrics
    """
    model.eval()
    tracker = MetricsTracker()
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            if isinstance(outputs, list):
                predictions = torch.sigmoid(outputs[0])
            else:
                predictions = torch.sigmoid(outputs)
            
            # Update metrics
            tracker.update(predictions, masks, threshold)
    
    # Get comprehensive results
    avg_metrics = tracker.get_average_metrics()
    best_metrics = tracker.get_best_metrics()
    
    # Combine results
    final_metrics = avg_metrics.copy()
    final_metrics.update(best_metrics)
    
    return final_metrics 