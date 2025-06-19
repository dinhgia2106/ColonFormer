"""
Evaluation Script cho ColonFormer
Đánh giá model trên các test datasets: Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-Larib
"""

import os
import argparse
import torch
import numpy as np
import cv2
from datetime import datetime

from models.colonformer import ColonFormer
from datasets import PolypDataModule
from utils.metrics import MetricTracker, evaluate_model, print_metrics


def load_model(model, checkpoint_path, device):
    """Load model từ checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        if 'best_dice' in checkpoint:
            print(f"Checkpoint best dice: {checkpoint['best_dice']:.4f}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print(f"Model loaded from state dict: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_single_dataset(model, dataset_name, data_module, device):
    """
    Đánh giá model trên một dataset cụ thể
    """
    print(f"\nEvaluating on {dataset_name}...")
    print("-" * 40)
    
    # Get test dataloader
    test_loader = data_module.get_test_dataloader(dataset_name)
    
    if test_loader is None:
        print(f"Warning: Không thể tạo test loader cho {dataset_name}")
        return None
    
    print(f"Number of test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print_metrics(metrics, f"{dataset_name}")
    
    return metrics


def save_results(all_results, save_path):
    """Lưu kết quả evaluation"""
    with open(save_path, 'w') as f:
        f.write("ColonFormer Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for dataset_name, metrics in all_results.items():
            if metrics is not None:
                f.write(f"\n{dataset_name}:\n")
                f.write("-" * 20 + "\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
            else:
                f.write(f"\n{dataset_name}: No data available\n")
        
        # Calculate overall averages (excluding None results)
        valid_results = {k: v for k, v in all_results.items() if v is not None}
        if valid_results:
            f.write("\nOverall Averages:\n")
            f.write("-" * 20 + "\n")
            
            # Average each metric across datasets
            all_metrics = {}
            for metrics in valid_results.values():
                for metric_name, value in metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
            
            for metric_name, values in all_metrics.items():
                avg_value = np.mean(values)
                f.write(f"  Average {metric_name}: {avg_value:.4f}\n")
    
    print(f"\nResults saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='ColonFormer Evaluation')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='mit_b3',
                        choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'],
                        help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='data',
                        help='Data root directory')
    parser.add_argument('--img_size', type=int, default=352,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data workers')
    
    # Evaluation parameters
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-Larib'],
                        help='Datasets to evaluate on')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save results')
    
    # Other
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating model: {args.backbone}")
    model = ColonFormer(
        backbone=args.backbone,
        num_classes=args.num_classes
    )
    
    # Load checkpoint
    model = load_model(model, args.checkpoint, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create data module
    print("Preparing data...")
    data_module = PolypDataModule(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Evaluate on each dataset
    all_results = {}
    
    print("\nStarting evaluation...")
    print("=" * 50)
    
    for dataset_name in args.datasets:
        try:
            metrics = evaluate_single_dataset(model, dataset_name, data_module, device)
            all_results[dataset_name] = metrics
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")
            all_results[dataset_name] = None
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 50)
    
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    
    if valid_results:
        # Print results table
        print(f"{'Dataset':<15} {'mDice':<8} {'mIoU':<8} {'Precision':<10} {'Recall':<8}")
        print("-" * 55)
        
        for dataset_name, metrics in valid_results.items():
            dice = metrics.get('mDice', 0)
            iou = metrics.get('mIoU', 0)
            precision = metrics.get('Precision', 0)
            recall = metrics.get('Recall', 0)
            
            print(f"{dataset_name:<15} {dice:<8.4f} {iou:<8.4f} {precision:<10.4f} {recall:<8.4f}")
        
        # Calculate and print averages
        print("-" * 55)
        all_metrics = {}
        for metrics in valid_results.values():
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        avg_dice = np.mean(all_metrics.get('mDice', [0]))
        avg_iou = np.mean(all_metrics.get('mIoU', [0]))
        avg_precision = np.mean(all_metrics.get('Precision', [0]))
        avg_recall = np.mean(all_metrics.get('Recall', [0]))
        
        print(f"{'Average':<15} {avg_dice:<8.4f} {avg_iou:<8.4f} {avg_precision:<10.4f} {avg_recall:<8.4f}")
    
    else:
        print("No valid results found!")
    
    # Save results
    if args.save_results:
        save_results(all_results, args.save_results)
    else:
        # Auto-generate save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"evaluation_results_{timestamp}.txt"
        save_results(all_results, save_path)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main() 