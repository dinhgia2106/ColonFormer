"""
Demo Experiment Management System
Showcase toàn bộ hệ thống experiment tracking và test management
"""

import os
import argparse
import json
import pandas as pd
from datetime import datetime

def demo_experiment_tracking():
    """Demo experiment tracking system"""
    print("Demo: Experiment Tracking System")
    print("="*60)
    
    try:
        from utils.experiment_tracker import ExperimentTracker
        
        # Create demo tracker
        tracker = ExperimentTracker("demo_experiments", "demo_colonformer")
        
        # Demo model config
        print("1. Logging Model Configuration...")
        dummy_model_config = {
            'name': 'ColonFormer-B3',
            'backbone': 'mit_b3',
            'total_parameters': 45_000_000,
            'num_classes': 1,
            'img_size': 352,
            'deep_supervision': True,
            'use_refinement': True
        }
        
        tracker.config['model'] = dummy_model_config
        
        # Demo training config
        print("2. Logging Training Configuration...")
        dummy_training_config = {
            'epochs': 20,
            'batch_size': 8,
            'lr': 1e-4,
            'optimizer': {
                'type': 'Adam',
                'params': {'betas': [0.9, 0.999], 'weight_decay': 1e-4}
            },
            'scheduler': {
                'type': 'CosineAnnealingLR',
                'T_max': 20
            },
            'criterion': {
                'type': 'ColonFormerLoss',
                'focal_alpha': 0.25,
                'focal_gamma': 2.0,
                'loss_lambda': 1.0
            }
        }
        
        tracker.config['training'] = dummy_training_config
        
        # Demo data config
        print("3. Logging Data Configuration...")
        dummy_data_config = {
            'data_root': 'data/TrainDataset',
            'img_size': 352,
            'batch_size': 8,
            'val_split': 0.2,
            'train_size': 1200,
            'val_size': 300
        }
        
        tracker.config['data'] = dummy_data_config
        
        # Save config
        tracker.save_config()
        
        # Demo epoch results
        print("4. Logging Training Progress...")
        import numpy as np
        
        for epoch in range(1, 6):
            # Simulate improving metrics
            train_loss = 0.5 * np.exp(-epoch * 0.2) + np.random.normal(0, 0.02)
            val_loss = 0.4 * np.exp(-epoch * 0.15) + np.random.normal(0, 0.02)
            
            train_dice = 0.6 + 0.3 * (1 - np.exp(-epoch * 0.3)) + np.random.normal(0, 0.01)
            val_dice = 0.65 + 0.25 * (1 - np.exp(-epoch * 0.25)) + np.random.normal(0, 0.01)
            
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
            
            tracker.log_epoch_results(epoch, train_metrics, val_metrics, lr, epoch_time)
            print(f"  Epoch {epoch}: Val Dice = {val_metrics['mDice']:.4f}")
        
        # Log final results
        tracker.log_final_results(best_checkpoint_path=os.path.join(tracker.exp_dir, 'best_model.pth'))
        
        # Print summary
        tracker.print_summary()
        
        print(f"\nExperiment files created in: {tracker.exp_dir}")
        print(f"Experiment ID: {tracker.experiment_id}")
        
        return tracker.experiment_id, tracker.exp_dir
        
    except ImportError as e:
        print(f"Import error: {e}")
        return None, None


def demo_test_management(experiment_id=None, experiment_dir=None):
    """Demo test management system"""
    print("\nDemo: Test Management System")
    print("="*60)
    
    try:
        from utils.test_manager import TestResultsManager
        
        # Create test manager
        manager = TestResultsManager("demo_experiments")
        
        if experiment_id and experiment_dir:
            print(f"1. Adding test results for experiment {experiment_id}...")
            
            # Demo test results for different datasets
            datasets = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-Larib']
            
            for dataset in datasets:
                # Simulate test results
                import numpy as np
                base_dice = 0.85 + np.random.normal(0, 0.05)
                base_iou = base_dice - 0.05 + np.random.normal(0, 0.02)
                
                test_results = {
                    'mDice': min(max(base_dice, 0.5), 0.95),
                    'mIoU': min(max(base_iou, 0.4), 0.9),
                    'Precision': min(max(base_dice + 0.02, 0.5), 0.95),
                    'Recall': min(max(base_dice - 0.01, 0.5), 0.95)
                }
                
                manager.add_test_results(
                    experiment_id=experiment_id,
                    test_results=test_results,
                    dataset_name=dataset,
                    notes=f"Demo test on {dataset}"
                )
                
                print(f"  {dataset}: mDice = {test_results['mDice']:.4f}")
        
        # Show summary
        print("\n2. Results Summary:")
        manager.print_summary()
        
        # Show CSV export
        print("\n3. CSV Export:")
        df = manager.get_summary_table()
        if not df.empty:
            print("Columns:", list(df.columns))
            print("Shape:", df.shape)
            print(f"CSV saved to: {manager.csv_file}")
        
        return manager
        
    except ImportError as e:
        print(f"Import error: {e}")
        return None


def demo_paper_table(manager):
    """Demo paper-style results table"""
    print("\nDemo: Paper-style Results Table")
    print("="*60)
    
    try:
        from utils.test_manager import create_paper_results_table
        
        # Create paper table
        paper_table = create_paper_results_table(
            manager, 
            output_file=os.path.join("demo_experiments", "demo_paper_results.csv")
        )
        
        if paper_table is not None:
            print("\nPaper-style Results Table:")
            print(paper_table.to_string(index=False))
        else:
            print("No results available for paper table")
        
    except Exception as e:
        print(f"Error creating paper table: {e}")


def show_usage_examples():
    """Show usage examples"""
    print("USAGE EXAMPLES - Experiment Management System")
    print("="*70)
    
    examples = [
        ("Training với logging:", [
            "python train.py --epochs 20 --backbone mit_b3 --batch_size 8",
            "# Tự động tạo experiment ID và log toàn bộ config"
        ]),
        
        ("List experiments:", [
            "python test_experiments.py --list_experiments",
            "# Hiển thị tất cả experiments và trạng thái"
        ]),
        
        ("List untested experiments:", [
            "python test_experiments.py --list_untested",
            "# Hiển thị experiments chưa được test"
        ]),
        
        ("Auto-test all untested:", [
            "python test_experiments.py",
            "# Tự động test tất cả experiments chưa test"
        ]),
        
        ("Test specific experiment:", [
            "python test_experiments.py --experiment_id abc12345",
            "# Test experiment cụ thể"
        ]),
        
        ("Test with prediction saving:", [
            "python test_experiments.py --save_predictions",
            "# Save prediction visualizations"
        ]),
        
        ("Show results summary:", [
            "python test_experiments.py --summary",
            "# Hiển thị tổng quan kết quả"
        ]),
        
        ("Create paper table:", [
            "python test_experiments.py --create_paper_table",
            "# Tạo bảng kết quả theo format paper"
        ]),
        
        ("Force retest:", [
            "python test_experiments.py --force",
            "# Force retest experiments đã test"
        ])
    ]
    
    for title, commands in examples:
        print(f"\n{title}")
        print("-" * len(title))
        for cmd in commands:
            if cmd.startswith("#"):
                print(f"    {cmd}")
            else:
                print(f"    $ {cmd}")
    
    print("\n" + "="*70)


def show_file_structure():
    """Show expected file structure"""
    print("FILE STRUCTURE AFTER TRAINING & TESTING")
    print("="*60)
    
    structure = """
checkpoints/
├── colonformer_mit_b3_abc12345/
│   ├── experiment_config.json       # Toàn bộ config
│   ├── results.json                 # Training results
│   ├── training.log                 # Training logs
│   ├── best_model.pth               # Best checkpoint
│   ├── logs/                        # Training history
│   │   ├── experiment_train.json
│   │   └── experiment_val.json
│   ├── plots/                       # Training plots
│   │   └── experiment_progress.png
│   └── predictions/                 # Test predictions (optional)
│       ├── Kvasir/
│       ├── CVC-ClinicDB/
│       └── ...
├── colonformer_mit_b1_def67890/
│   └── ...
├── test_results_summary.json        # Consolidated test results
├── test_results_summary.csv         # CSV export
└── paper_results.csv               # Paper-style table
    """
    
    print(structure)
    
    print("Key Features:")
    print("- Mỗi experiment có unique ID (8 ký tự)")
    print("- Log đầy đủ config: model, training, data")
    print("- Auto-detect experiments chưa test")
    print("- Consolidated results trong CSV")
    print("- Paper-style table như trong bài báo")


def cleanup_demo():
    """Cleanup demo files"""
    import shutil
    
    if os.path.exists("demo_experiments"):
        shutil.rmtree("demo_experiments")
        print("Demo files cleaned up")


def main():
    parser = argparse.ArgumentParser(description='Demo Experiment Management System')
    parser.add_argument('--mode', choices=['demo', 'usage', 'structure', 'cleanup'], 
                        default='demo',
                        help='Demo mode')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # Run full demo
        exp_id, exp_dir = demo_experiment_tracking()
        manager = demo_test_management(exp_id, exp_dir)
        if manager:
            demo_paper_table(manager)
    
    elif args.mode == 'usage':
        show_usage_examples()
    
    elif args.mode == 'structure':
        show_file_structure()
    
    elif args.mode == 'cleanup':
        cleanup_demo()
    
    print(f"\nRun other demos:")
    print(f"python demo_experiment_system.py --mode usage      # Usage examples")
    print(f"python demo_experiment_system.py --mode structure  # File structure")
    print(f"python demo_experiment_system.py --mode cleanup    # Cleanup demo files")


if __name__ == "__main__":
    main() 