"""
Evaluation metrics cho Polyp Segmentation
Bao gồm: mDice, mIoU, Recall, Precision
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Tính Dice coefficient
    Args:
        pred: prediction tensor (B, 1, H, W) hoặc (B, H, W)
        target: ground truth tensor (B, 1, H, W) hoặc (B, H, W)
        smooth: smoothing factor
    Returns:
        dice: Dice coefficient
    """
    # Ensure tensors are on same device
    pred = pred.float()
    target = target.float()
    
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate intersection
    intersection = (pred_flat * target_flat).sum()
    
    # Calculate Dice
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice


def iou_coefficient(pred, target, smooth=1e-6):
    """
    Tính IoU (Intersection over Union)
    Args:
        pred: prediction tensor (B, 1, H, W) hoặc (B, H, W)
        target: ground truth tensor (B, 1, H, W) hoặc (B, H, W)
        smooth: smoothing factor
    Returns:
        iou: IoU coefficient
    """
    # Ensure tensors are on same device
    pred = pred.float()
    target = target.float()
    
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def precision_recall(pred, target):
    """
    Tính Precision và Recall
    Args:
        pred: prediction tensor (B, 1, H, W) hoặc (B, H, W)
        target: ground truth tensor (B, 1, H, W) hoặc (B, H, W)
    Returns:
        precision: Precision score
        recall: Recall score
    """
    # Convert to numpy
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    
    # Convert to binary
    pred_binary = (pred_np > 0.5).astype(int)
    target_binary = (target_np > 0.5).astype(int)
    
    # Calculate precision and recall
    precision = precision_score(target_binary, pred_binary, zero_division=0)
    recall = recall_score(target_binary, pred_binary, zero_division=0)
    
    return precision, recall


class MetricTracker:
    """
    Class để theo dõi metrics qua các epochs
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tất cả metrics"""
        self.dice_scores = []
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.losses = []
    
    def update(self, pred, target, loss=None):
        """
        Update metrics với prediction mới
        Args:
            pred: prediction tensor
            target: ground truth tensor
            loss: loss value (optional)
        """
        # Apply sigmoid if needed (for logits)
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
        
        # Apply threshold
        pred_binary = (pred > 0.5).float()
        
        # Calculate metrics
        dice = dice_coefficient(pred_binary, target)
        iou = iou_coefficient(pred_binary, target)
        precision, recall = precision_recall(pred_binary, target)
        
        # Store metrics
        self.dice_scores.append(dice.item())
        self.iou_scores.append(iou.item())
        self.precision_scores.append(precision)
        self.recall_scores.append(recall)
        
        if loss is not None:
            self.losses.append(loss)
    
    def get_average_metrics(self):
        """
        Tính trung bình của tất cả metrics
        Returns:
            dict: Dictionary chứa các metrics trung bình
        """
        metrics = {
            'mDice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'mIoU': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'Precision': np.mean(self.precision_scores) if self.precision_scores else 0.0,
            'Recall': np.mean(self.recall_scores) if self.recall_scores else 0.0,
        }
        
        if self.losses:
            metrics['Loss'] = np.mean(self.losses)
        
        return metrics
    
    def get_current_metrics(self):
        """
        Lấy metrics của batch cuối cùng
        Returns:
            dict: Dictionary chứa metrics hiện tại
        """
        if not self.dice_scores:
            return {}
        
        metrics = {
            'Dice': self.dice_scores[-1],
            'IoU': self.iou_scores[-1],
            'Precision': self.precision_scores[-1],
            'Recall': self.recall_scores[-1],
        }
        
        if self.losses:
            metrics['Loss'] = self.losses[-1]
        
        return metrics


def evaluate_model(model, dataloader, device, criterion=None):
    """
    Đánh giá model trên validation/test set
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: device để chạy
        criterion: loss function (optional)
    Returns:
        dict: Dictionary chứa các metrics
    """
    model.eval()
    metric_tracker = MetricTracker()
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Handle multiple outputs (deep supervision)
            if isinstance(outputs, (list, tuple)):
                # Take the final output for evaluation
                pred = outputs[0]
            else:
                pred = outputs
            
            # Calculate loss if criterion provided
            loss = None
            if criterion is not None:
                if isinstance(outputs, (list, tuple)):
                    # Deep supervision loss
                    loss_total = 0
                    for output in outputs:
                        loss_total += criterion(output, masks)
                    loss = loss_total / len(outputs)
                else:
                    loss = criterion(pred, masks)
                loss = loss.item()
            
            # Update metrics
            metric_tracker.update(pred, masks, loss)
    
    return metric_tracker.get_average_metrics()


def print_metrics(metrics, prefix=""):
    """
    In metrics một cách đẹp mắt
    Args:
        metrics: Dictionary chứa metrics
        prefix: Prefix cho output (ví dụ: "Train", "Val")
    """
    if prefix:
        prefix += " "
    
    output_lines = []
    for key, value in metrics.items():
        if key == 'Loss':
            output_lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            output_lines.append(f"{prefix}{key}: {value:.4f}")
    
    print(" | ".join(output_lines))


if __name__ == "__main__":
    # Test metrics
    import torch
    
    # Create dummy data
    pred = torch.rand(2, 1, 352, 352)
    target = torch.randint(0, 2, (2, 1, 352, 352)).float()
    
    # Test individual metrics
    dice = dice_coefficient(pred, target)
    iou = iou_coefficient(pred, target)
    precision, recall = precision_recall(pred, target)
    
    print(f"Dice: {dice:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Test metric tracker
    tracker = MetricTracker()
    tracker.update(pred, target, loss=0.5)
    
    metrics = tracker.get_average_metrics()
    print_metrics(metrics, "Test")
    
    print("Metrics test completed!") 