"""
Loss functions for ColonFormer training
Implementation based on the ColonFormer paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits from model [B, 1, H, W]
            targets: ground truth masks [B, 1, H, W]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss(nn.Module):
    """IoU Loss for better boundary prediction"""
    def __init__(self, smooth=1e-7):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: sigmoid probabilities [B, 1, H, W]
            targets: ground truth masks [B, 1, H, W]
        """
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: sigmoid probabilities [B, 1, H, W]
            targets: ground truth masks [B, 1, H, W]
        """
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class WeightedFocalIoULoss(nn.Module):
    """Combined Weighted Focal Loss and IoU Loss as used in ColonFormer"""
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, iou_weight=5.0, 
                 focal_weight=1.0, use_boundary_weight=True):
        super(WeightedFocalIoULoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.iou_loss = IoULoss()
        self.iou_weight = iou_weight
        self.focal_weight = focal_weight
        self.use_boundary_weight = use_boundary_weight
        
    def get_boundary_weight(self, targets, kernel_size=31):
        """Calculate boundary-aware weights"""
        # Apply average pooling to get smoothed version
        pooled = F.avg_pool2d(targets, kernel_size=kernel_size, stride=1, 
                             padding=kernel_size//2)
        
        # Calculate weight based on difference from smoothed version
        weight = 1 + 5 * torch.abs(pooled - targets)
        return weight
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits from model [B, 1, H, W]
            targets: ground truth masks [B, 1, H, W]
        """
        if self.use_boundary_weight:
            # Calculate boundary weights
            weight = self.get_boundary_weight(targets)
            
            # Weighted focal loss
            focal_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-focal_loss)
            focal_loss = self.focal_loss.alpha * (1 - pt) ** self.focal_loss.gamma * focal_loss
            weighted_focal_loss = (focal_loss * weight).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
            weighted_focal_loss = weighted_focal_loss.mean()
            
            # Weighted IoU loss
            pred = torch.sigmoid(inputs)
            inter = ((pred * targets) * weight).sum(dim=(2, 3))
            union = ((pred + targets) * weight).sum(dim=(2, 3))
            wiou = 1 - (inter + 1) / (union - inter + 1)
            wiou_loss = wiou.mean()
            
            total_loss = self.focal_weight * weighted_focal_loss + self.iou_weight * wiou_loss
        else:
            # Standard focal and IoU loss
            focal_loss = self.focal_loss(inputs, targets)
            iou_loss = self.iou_loss(inputs, targets)
            total_loss = self.focal_weight * focal_loss + self.iou_weight * iou_loss
            
        return total_loss


class StructureLoss(nn.Module):
    """Structure Loss from the original ColonFormer paper"""
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):
        super(StructureLoss, self).__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def forward(self, pred, mask):
        """
        Args:
            pred: prediction logits [B, 1, H, W]
            mask: ground truth masks [B, 1, H, W]
        """
        # Calculate boundary weight
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        
        # Weighted focal loss
        focal_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        pt = torch.exp(-focal_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * focal_loss
        wfocal = (focal_loss * weit).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        
        # Weighted IoU loss  
        pred_sigmoid = torch.sigmoid(pred)
        inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
        union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        
        return (wfocal + wiou).mean()


class EdgeAwareLoss(nn.Module):
    """Edge-aware loss for better boundary prediction"""
    def __init__(self, edge_weight=2.0):
        super(EdgeAwareLoss, self).__init__()
        self.edge_weight = edge_weight
        
        # Sobel kernels for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
    def get_edges(self, x):
        """Extract edges using Sobel operator"""
        if self.sobel_x.device != x.device:
            self.sobel_x = self.sobel_x.to(x.device)
            self.sobel_y = self.sobel_y.to(x.device)
            
        # Apply Sobel operators
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        
        # Calculate edge magnitude
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edges
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction logits [B, 1, H, W]
            targets: ground truth masks [B, 1, H, W]
        """
        pred = torch.sigmoid(inputs)
        
        # Get edges
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(targets)
        
        # Edge loss
        edge_loss = F.mse_loss(pred_edges, target_edges)
        
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        
        return bce_loss + self.edge_weight * edge_loss


class MultiScaleLoss(nn.Module):
    """Multi-scale loss for deep supervision"""
    def __init__(self, loss_fn, weights=None):
        super(MultiScaleLoss, self).__init__()
        self.loss_fn = loss_fn
        self.weights = weights or [1.0, 0.5, 0.25, 0.125]
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: list of predictions at different scales
            targets: ground truth masks
        """
        total_loss = 0
        
        for i, pred in enumerate(predictions):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            
            # Resize target to match prediction if needed
            if pred.shape[2:] != targets.shape[2:]:
                target_resized = F.interpolate(targets, size=pred.shape[2:], 
                                             mode='bilinear', align_corners=False)
            else:
                target_resized = targets
                
            loss = self.loss_fn(pred, target_resized)
            total_loss += weight * loss
            
        return total_loss 