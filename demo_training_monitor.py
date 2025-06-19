"""
Demo Training Monitor - Showcase Progress Bars và Real-time Visualization
"""

import argparse
import os
import sys

def demo_quick_training():
    """Demo quick training với 2 epochs"""
    print("Demo: Quick Training với Progress Monitoring")
    print("="*60)
    
    # Import training components
    try:
        from train import main as train_main
        import sys
        
        # Setup demo arguments
        demo_args = [
            '--epochs', '2',
            '--batch_size', '4', 
            '--backbone', 'mit_b1',
            '--data_root', 'data/TrainDataset',  # Sử dụng TrainDataset có sẵn
            '--save_dir', 'demo_checkpoints',
            '--log_interval', '5',
            '--val_split', '0.3'
        ]
        
        # Override sys.argv for demo
        original_argv = sys.argv.copy()
        sys.argv = ['train.py'] + demo_args
        
        print("Starting demo training với arguments:")
        for i in range(0, len(demo_args), 2):
            print(f"  {demo_args[i]}: {demo_args[i+1]}")
        print()
        
        # Run training
        train_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print("\nDemo completed! Check demo_checkpoints/ folder cho logs và plots.")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Đảm bảo tất cả dependencies đã được install.")
    except Exception as e:
        print(f"Demo error: {e}")


def demo_logger_visualization():
    """Demo logger và visualization features"""
    print("\nDemo: Training Logger và Visualization")
    print("="*60)
    
    try:
        from utils.logger import TrainingLogger
        import numpy as np
        import time
        
        # Create demo logger
        logger = TrainingLogger("demo_logs", "demo_experiment")
        
        # Simulate training data
        print("Generating demo training data...")
        epochs = 5
        
        for epoch in range(1, epochs + 1):
            # Simulate improving metrics
            train_loss = 0.5 * np.exp(-epoch * 0.2) + np.random.normal(0, 0.02)
            val_loss = 0.4 * np.exp(-epoch * 0.15) + np.random.normal(0, 0.02)
            
            train_dice = 0.6 + 0.3 * (1 - np.exp(-epoch * 0.3)) + np.random.normal(0, 0.01)
            val_dice = 0.65 + 0.25 * (1 - np.exp(-epoch * 0.25)) + np.random.normal(0, 0.01)
            
            # Create metric dictionaries
            train_metrics = {
                'Loss': max(train_loss, 0.01),
                'Dice': min(max(train_dice, 0), 1),
                'IoU': min(max(train_dice - 0.05, 0), 1),
                'Precision': min(max(train_dice + 0.02, 0), 1),
                'Recall': min(max(train_dice - 0.01, 0), 1)
            }
            
            val_metrics = {
                'Loss': max(val_loss, 0.01),
                'mDice': min(max(val_dice, 0), 1),
                'mIoU': min(max(val_dice - 0.03, 0), 1),
                'Precision': min(max(val_dice + 0.01, 0), 1),
                'Recall': min(max(val_dice - 0.02, 0), 1)
            }
            
            lr = 1e-4 * (0.5 ** (epoch / 3))
            epoch_time = 45 + np.random.normal(0, 5)
            
            # Log epoch
            logger.log_epoch(epoch, train_metrics, val_metrics, lr, epoch_time)
            
            print(f"Epoch {epoch}: Train Dice={train_metrics['Dice']:.4f}, "
                  f"Val Dice={val_metrics['mDice']:.4f}")
            
            time.sleep(0.5)  # Small delay for demo
        
        # Print summary
        logger.print_summary()
        
        print(f"\nDemo plots được tạo tại: {logger.plots_dir}")
        
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Demo error: {e}")


def show_features():
    """Show các tính năng của training monitor"""
    print("TÍNH NĂNG TRAINING MONITOR")
    print("="*60)
    
    features = [
        "1. Progress Bars với tqdm:",
        "   - Real-time progress cho từng epoch và batch",
        "   - Hiển thị ETA (estimated time arrival)",
        "   - Live metrics: Loss, Dice, IoU, Learning Rate",
        "   - Memory usage (CPU và GPU)",
        "   - Batch processing time",
        "",
        "2. Training Logger:",
        "   - Automatic logging metrics to JSON files",
        "   - Real-time plot generation",
        "   - Training history visualization",
        "   - Comprehensive training summary",
        "",
        "3. Metrics Tracking:",
        "   - Train/Validation Loss",
        "   - Dice Score và IoU",
        "   - Precision và Recall",
        "   - Learning Rate schedule",
        "   - Time per epoch",
        "",
        "4. Visualization:",
        "   - 6-panel training plots",
        "   - Automatic plot updates",
        "   - Comparison plots cho multiple runs",
        "   - High-resolution PNG exports",
        "",
        "5. Real-time Monitoring:",
        "   - Live progress updates",
        "   - Best model tracking",
        "   - Checkpoint management",
        "   - Training time estimates"
    ]
    
    for feature in features:
        print(feature)
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Demo Training Monitor')
    parser.add_argument('--mode', choices=['training', 'logger', 'features'], 
                        default='features',
                        help='Demo mode')
    
    args = parser.parse_args()
    
    if args.mode == 'training':
        demo_quick_training()
    elif args.mode == 'logger':
        demo_logger_visualization()
    else:
        show_features()
        print("\nChạy demo cụ thể:")
        print("python demo_training_monitor.py --mode training   # Demo full training")
        print("python demo_training_monitor.py --mode logger     # Demo logger only")


if __name__ == "__main__":
    main() 