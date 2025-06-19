"""
Loss Functions cho ColonFormer
Bao gồm Weighted Focal Loss và Weighted IoU Loss như mô tả trong bài báo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


def compute_distance_weights(masks, kernel_size=31):
    """
    Tính toán ma trận trọng số beta dựa trên khoảng cách đến biên polyp
    
    Args:
        masks: ground truth masks [B, 1, H, W]
        kernel_size: kích thước kernel cho morphological operations
    
    Returns:
        beta: weight matrix [B, 1, H, W]
    """
    device = masks.device
    batch_size, _, height, width = masks.shape
    beta = torch.zeros_like(masks, dtype=torch.float32)
    
    # Tạo structuring element
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    for b in range(batch_size):
        mask = masks[b, 0].cpu().numpy()  # [H, W]
        
        if mask.max() > 0:  # Có polyp trong mask
            # Tính toán khoảng cách đến biên
            # Distance transform for foreground pixels
            fg_distances = ndimage.distance_transform_edt(mask > 0.5)
            
            # Distance transform for background pixels
            bg_distances = ndimage.distance_transform_edt(mask <= 0.5)
            
            # Combine distances
            distances = np.where(mask > 0.5, fg_distances, -bg_distances)
            
            # Compute weights (importance decreases with distance from boundary)
            max_dist = max(fg_distances.max(), bg_distances.max())
            if max_dist > 0:
                weights = 1.0 + np.exp(-np.abs(distances) / (max_dist / 3.0))
            else:
                weights = np.ones_like(mask)
        else:
            # Không có polyp, trọng số uniform
            weights = np.ones_like(mask)
        
        beta[b, 0] = torch.from_numpy(weights).to(device)
    
    return beta


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss như mô tả trong công thức (3) của bài báo
    """
    def __init__(self, alpha=0.25, gamma=2.0, weight_computation='distance'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight_computation = weight_computation  # 'distance' hoặc 'simple'
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: predicted probabilities [B, 1, H, W]
            targets: ground truth labels [B, 1, H, W] (0 hoặc 1)
        
        Returns:
            weighted_focal_loss: scalar loss value
        """
        # Ensure predictions are probabilities
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute focal loss components
        ce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Weighted focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return focal_loss.mean()


class WeightedIoULoss(nn.Module):
    """
    Weighted IoU Loss như mô tả trong công thức (4) của bài báo
    """
    def __init__(self, smooth=1e-6, weight_lambda=1.0):
        super(WeightedIoULoss, self).__init__()
        self.smooth = smooth
        self.weight_lambda = weight_lambda
        
    def forward(self, predictions, targets, importance_weights=None):
        """
        Args:
            predictions: predicted probabilities [B, 1, H, W]
            targets: ground truth labels [B, 1, H, W]
            importance_weights: beta weights [B, 1, H, W]
            
        Returns:
            weighted_iou_loss: scalar loss value
        """
        # Ensure predictions are probabilities
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = torch.sigmoid(predictions)
        
        # Compute importance weights if not provided
        if importance_weights is None:
            importance_weights = compute_distance_weights(targets)
        
        # Weighted intersection and union
        intersection = (predictions * targets * importance_weights).sum(dim=(2, 3))
        union = ((predictions + targets) * importance_weights).sum(dim=(2, 3)) - intersection
        
        # IoU computation
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # IoU loss (1 - IoU)
        iou_loss = 1.0 - iou.mean()
        
        return iou_loss


class ColonFormerLoss(nn.Module):
    """
    Combined loss function cho ColonFormer
    L_total = lambda * L_wfocal + L_wiou
    """
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, iou_smooth=1e-6, 
                 loss_lambda=1.0, deep_supervision=True):
        super(ColonFormerLoss, self).__init__()
        self.focal_loss = WeightedFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.iou_loss = WeightedIoULoss(smooth=iou_smooth)
        self.loss_lambda = loss_lambda
        self.deep_supervision = deep_supervision
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: có thể là:
                - single prediction [B, 1, H, W]
                - dict với 'main' và 'aux' predictions cho deep supervision
            targets: ground truth masks [B, 1, H, W]
            
        Returns:
            total_loss: combined loss
            loss_dict: dictionary với breakdown của losses
        """
        loss_dict = {}
        
        # Handle different prediction formats
        if isinstance(predictions, dict):
            main_pred = predictions['main']
            aux_preds = predictions.get('aux', [])
        elif isinstance(predictions, (list, tuple)):
            main_pred = predictions[0]
            aux_preds = predictions[1:] if len(predictions) > 1 else []
        else:
            main_pred = predictions
            aux_preds = []
        
        # Resize predictions to match target size if needed
        target_size = targets.shape[2:]
        if main_pred.shape[2:] != target_size:
            main_pred = F.interpolate(
                main_pred, size=target_size, mode='bilinear', align_corners=True
            )
        
        # Compute importance weights
        importance_weights = compute_distance_weights(targets)
        
        # Main loss computation
        focal_loss_main = self.focal_loss(main_pred, targets)
        iou_loss_main = self.iou_loss(main_pred, targets, importance_weights)
        main_loss = self.loss_lambda * focal_loss_main + iou_loss_main
        
        loss_dict['focal_main'] = focal_loss_main
        loss_dict['iou_main'] = iou_loss_main
        loss_dict['main'] = main_loss
        
        total_loss = main_loss
        
        # Deep supervision losses
        if self.deep_supervision and aux_preds:
            aux_loss_total = 0.0
            for i, aux_pred in enumerate(aux_preds):
                # Resize auxiliary prediction
                if aux_pred.shape[2:] != target_size:
                    aux_pred = F.interpolate(
                        aux_pred, size=target_size, mode='bilinear', align_corners=True
                    )
                
                focal_loss_aux = self.focal_loss(aux_pred, targets)
                iou_loss_aux = self.iou_loss(aux_pred, targets, importance_weights)
                aux_loss = self.loss_lambda * focal_loss_aux + iou_loss_aux
                
                # Weight auxiliary losses (giảm dần theo độ sâu)
                aux_weight = 0.4 / (i + 1)  # 0.4, 0.2, 0.133, ...
                weighted_aux_loss = aux_weight * aux_loss
                aux_loss_total += weighted_aux_loss
                
                loss_dict[f'focal_aux_{i}'] = focal_loss_aux
                loss_dict[f'iou_aux_{i}'] = iou_loss_aux
                loss_dict[f'aux_{i}'] = aux_loss
            
            loss_dict['aux_total'] = aux_loss_total
            total_loss = total_loss + aux_loss_total
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


class DiceLoss(nn.Module):
    """
    Dice Loss như một alternative loss
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


if __name__ == "__main__":
    # Test loss functions
    batch_size, channels, height, width = 2, 1, 128, 128
    
    # Tạo dummy data
    predictions = torch.randn(batch_size, channels, height, width)
    targets = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    # Test individual losses
    focal_loss = WeightedFocalLoss()
    iou_loss = WeightedIoULoss()
    dice_loss = DiceLoss()
    
    focal_val = focal_loss(predictions, targets)
    iou_val = iou_loss(predictions, targets)
    dice_val = dice_loss(predictions, targets)
    
    print(f"Focal Loss: {focal_val.item():.4f}")
    print(f"IoU Loss: {iou_val.item():.4f}")
    print(f"Dice Loss: {dice_val.item():.4f}")
    
    # Test combined loss
    combined_loss = ColonFormerLoss()
    
    # Test với single prediction
    total_loss, loss_dict = combined_loss(predictions, targets)
    print(f"\nCombined Loss: {total_loss.item():.4f}")
    print("Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test với deep supervision
    aux_preds = [
        torch.randn(batch_size, channels, height//2, width//2),
        torch.randn(batch_size, channels, height//4, width//4)
    ]
    
    predictions_dict = {
        'main': predictions,
        'aux': aux_preds
    }
    
    total_loss_ds, loss_dict_ds = combined_loss(predictions_dict, targets)
    print(f"\nDeep Supervision Loss: {total_loss_ds.item():.4f}")
    print("Deep supervision loss breakdown:")
    for key, value in loss_dict_ds.items():
        print(f"  {key}: {value.item():.4f}") 