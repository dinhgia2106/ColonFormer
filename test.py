#!/usr/bin/env python
"""
ColonFormer Test Script - Optimized Version
Test toàn bộ data trong TestDataset với metrics Dice, IoU, Precision, Recall
Sử dụng code đã optimize từ train.py
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Import từ train.py đã optimize
from train import ColonFormer, Config, calculate_metrics


class TestDataset(Dataset):
    """Dataset cho testing trên toàn bộ TestDataset"""
    
    def __init__(self, test_root, img_size=352):
        self.test_root = Path(test_root)
        self.img_size = img_size
        
        # Load tất cả các dataset con
        self.datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
        self.data_samples = []
        
        for dataset_name in self.datasets:
            dataset_path = self.test_root / dataset_name
            if not dataset_path.exists():
                print(f"Warning: Dataset {dataset_name} not found at {dataset_path}")
                continue
                
            img_dir = dataset_path / 'images'
            mask_dir = dataset_path / 'masks'
            
            if not img_dir.exists() or not mask_dir.exists():
                print(f"Warning: Images or masks folder not found for {dataset_name}")
                continue
            
            # Load all images và masks
            img_paths = sorted(list(img_dir.glob('*.png')))
            mask_paths = sorted(list(mask_dir.glob('*.png')))
            
            if len(img_paths) != len(mask_paths):
                print(f"Warning: Mismatch between images and masks in {dataset_name}")
                continue
            
            for img_path, mask_path in zip(img_paths, mask_paths):
                self.data_samples.append({
                    'image': img_path,
                    'mask': mask_path,
                    'dataset': dataset_name
                })
        
        print(f"Total test samples: {len(self.data_samples)}")
        for dataset_name in self.datasets:
            count = sum(1 for sample in self.data_samples if sample['dataset'] == dataset_name)
            if count > 0:
                print(f"  {dataset_name}: {count} samples")

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # Load image
        img = cv2.imread(str(sample['image']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = img.shape[:2]  # (H, W)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Load mask
        mask = cv2.imread(str(sample['mask']), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 127).astype(np.float32)
        
        # Convert to tensors - CRITICAL: ensure float32 type
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC -> CHW, ensure float32
        mask = torch.from_numpy(mask).unsqueeze(0).float()    # Add channel dim, ensure float32
        
        return img, mask, sample['dataset'], original_size


def test_model(model, test_loader, device, config):
    """Test model trên test dataset"""
    model.eval()
    
    # Metrics cho từng dataset và overall
    dataset_metrics = {}
    overall_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for images, masks, dataset_names, original_sizes in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get main output nếu có multi-scale
            if isinstance(outputs, (list, tuple)):
                main_output = outputs[0]
            else:
                main_output = outputs
            
            # Resize output về original mask size
            main_output_resized = F.interpolate(main_output, size=masks.shape[2:], mode='bilinear', align_corners=False)
            
            # Calculate metrics cho từng sample
            for i in range(images.size(0)):
                pred = main_output_resized[i:i+1]
                mask = masks[i:i+1]
                dataset_name = dataset_names[i]
                
                metrics = calculate_metrics(pred, mask)
                
                # Update dataset-specific metrics
                if dataset_name not in dataset_metrics:
                    dataset_metrics[dataset_name] = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'count': 0}
                
                for key in ['dice', 'iou', 'precision', 'recall']:
                    dataset_metrics[dataset_name][key] += metrics[key]
                    overall_metrics[key] += metrics[key]
                
                dataset_metrics[dataset_name]['count'] += 1
                total_samples += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Dice': f'{metrics["dice"]:.4f}',
                    'IoU': f'{metrics["iou"]:.4f}',
                    'Dataset': dataset_name
                })
    
    # Calculate averages
    for dataset_name in dataset_metrics:
        count = dataset_metrics[dataset_name]['count']
        for key in ['dice', 'iou', 'precision', 'recall']:
            dataset_metrics[dataset_name][key] /= count
    
    for key in overall_metrics:
        overall_metrics[key] /= total_samples
    
    return overall_metrics, dataset_metrics


def load_model_and_config(checkpoint_path):
    """Load model và config từ checkpoint"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Tạo config từ checkpoint hoặc từ default
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        
        # Convert dict thành args object để tạo Config
        class Args:
            pass
        
        args = Args()
        for key, value in config_dict.items():
            # Fix key mapping issues
            if key == 'learning_rate':
                setattr(args, 'lr', value)
            elif key == 'optimizer_type':
                setattr(args, 'optimizer', value)
            elif key == 'scheduler_type':
                setattr(args, 'scheduler', value)
            else:
                setattr(args, key, value)
        
        # Ensure all required attributes exist
        required_attrs = ['backbone', 'use_refinement', 'num_classes', 'epochs', 
                         'batch_size', 'lr', 'weight_decay', 'img_size', 'loss_type',
                         'focal_alpha', 'focal_gamma', 'iou_weight', 'optimizer', 
                         'scheduler', 'momentum', 'data_root', 'val_split', 
                         'num_workers', 'work_dir', 'resume_from', 'seed']
        
        for attr in required_attrs:
            if not hasattr(args, attr):
                # Set default values
                defaults = {
                    'backbone': 'simple', 'use_refinement': True, 'num_classes': 1,
                    'epochs': 100, 'batch_size': 4, 'lr': 1e-4, 'weight_decay': 1e-4,
                    'img_size': 352, 'loss_type': 'structure', 'focal_alpha': 0.25,
                    'focal_gamma': 2.0, 'iou_weight': 1.0, 'optimizer': 'adamw',
                    'scheduler': 'cosine', 'momentum': 0.9, 'data_root': 'data/TrainDataset',
                    'val_split': 0.2, 'num_workers': 2, 'work_dir': 'work_dirs/colonformer',
                    'resume_from': None, 'seed': 42
                }
                setattr(args, attr, defaults.get(attr, None))
        
        config = Config(args)
        else:
        # Default config nếu không có trong checkpoint
        class Args:
            backbone = 'simple'
            use_refinement = True
            num_classes = 1
            epochs = 100
            batch_size = 4
            lr = 1e-4
            weight_decay = 1e-4
            img_size = 352
            loss_type = 'structure'
            focal_alpha = 0.25
            focal_gamma = 2.0
            iou_weight = 1.0
            optimizer = 'adamw'
            scheduler = 'cosine'
            momentum = 0.9
            data_root = 'data/TrainDataset'
            val_split = 0.2
            num_workers = 2
            work_dir = 'work_dirs/colonformer'
            resume_from = None
            seed = 42
            
        config = Config(Args())
    
    # Tạo model
    model = ColonFormer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config, checkpoint


def print_results(overall_metrics, dataset_metrics, config, checkpoint_info):
    """In kết quả test với config details"""
    print("\n" + "=" * 80)
    print("COLONFORMER TEST RESULTS")
    print("=" * 80)
    
    # Print config summary
    print("\nCONFIGURATION SUMMARY:")
    print(f"  Model: ColonFormer (backbone: {config.backbone})")
    print(f"  Refinement: {'Enabled' if config.use_refinement else 'Disabled'}")
    print(f"  Loss Function: {config.loss_type}")
    print(f"  Image Size: {config.img_size}x{config.img_size}")
    if hasattr(config, 'learning_rate'):
        print(f"  Learning Rate: {config.learning_rate}")
    if hasattr(config, 'batch_size'):
        print(f"  Batch Size: {config.batch_size}")
    
    # Print checkpoint info
    if 'epoch' in checkpoint_info:
        print(f"  Checkpoint Epoch: {checkpoint_info['epoch']}")
    if 'best_dice' in checkpoint_info:
        print(f"  Best Training Dice: {checkpoint_info['best_dice']:.4f}")
    
    # Print overall results
    print(f"\nOVERALL RESULTS:")
    print(f"  Dice Score:  {overall_metrics['dice']:.4f}")
    print(f"  IoU Score:   {overall_metrics['iou']:.4f}")
    print(f"  Precision:   {overall_metrics['precision']:.4f}")
    print(f"  Recall:      {overall_metrics['recall']:.4f}")
    
    # Print per-dataset results
    print(f"\nPER-DATASET RESULTS:")
    for dataset_name, metrics in dataset_metrics.items():
        print(f"\n  {dataset_name} ({metrics['count']} samples):")
        print(f"    Dice:      {metrics['dice']:.4f}")
        print(f"    IoU:       {metrics['iou']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
    
    print("\n" + "=" * 80)


def save_results(overall_metrics, dataset_metrics, config, checkpoint_info, output_path):
    """Lưu kết quả test vào file JSON"""
    results = {
        'test_info': {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'checkpoint_epoch': checkpoint_info.get('epoch', 'unknown'),
            'best_training_dice': checkpoint_info.get('best_dice', 'unknown')
        },
        'config': {
            'backbone': config.backbone,
            'use_refinement': config.use_refinement,
            'loss_type': config.loss_type,
            'img_size': config.img_size,
            'learning_rate': getattr(config, 'learning_rate', 'unknown'),
            'batch_size': getattr(config, 'batch_size', 'unknown')
        },
        'overall_metrics': overall_metrics,
        'dataset_metrics': dataset_metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ColonFormer Test Script - Optimized Version')
    
    # Test configs
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--test-data', default='data/TestDataset', help='Path to test dataset')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=8, help='Test batch size')
    parser.add_argument('--img-size', type=int, default=352, help='Input image size')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Test data: {args.test_data}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load model và config
    print("Loading model và config...")
    try:
        model, config, checkpoint = load_model_and_config(args.checkpoint)
        model = model.to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Override img_size từ args nếu khác
    if args.img_size != config.img_size:
        print(f"Overriding image size: {config.img_size} -> {args.img_size}")
        config.img_size = args.img_size
    
    # Load test dataset
    print("Loading test dataset...")
    try:
        test_dataset = TestDataset(args.test_data, args.img_size)
        if len(test_dataset) == 0:
            print("No test samples found!")
            return
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Run test
    print("Starting test...")
    overall_metrics, dataset_metrics = test_model(model, test_loader, device, config)
    
    # Print results
    print_results(overall_metrics, dataset_metrics, config, checkpoint)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"test_results_{timestamp}.json"
    save_results(overall_metrics, dataset_metrics, config, checkpoint, results_file)
    
    print(f"\nTest completed successfully!")


if __name__ == '__main__':
    main() 