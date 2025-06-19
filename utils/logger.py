"""
Training Logger và Visualization utilities
"""

import os
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


class TrainingLogger:
    """
    Logger để track training progress và tạo visualization
    """
    
    def __init__(self, save_dir, experiment_name=None):
        self.save_dir = save_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.log_dir = os.path.join(save_dir, 'logs')
        self.plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize logs
        self.train_history = {
            'epoch': [],
            'loss': [],
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'lr': [],
            'time': []
        }
        
        self.val_history = {
            'epoch': [],
            'loss': [],
            'mdice': [],
            'miou': [],
            'precision': [],
            'recall': []
        }
        
        self.epoch_times = []
        self.start_time = time.time()
        
        # Log file paths
        self.train_log_path = os.path.join(self.log_dir, f'{self.experiment_name}_train.json')
        self.val_log_path = os.path.join(self.log_dir, f'{self.experiment_name}_val.json')
    
    def log_epoch(self, epoch, train_metrics, val_metrics, lr, epoch_time):
        """Log một epoch"""
        # Train metrics
        self.train_history['epoch'].append(epoch)
        self.train_history['loss'].append(train_metrics['Loss'])
        self.train_history['dice'].append(train_metrics['Dice'])
        self.train_history['iou'].append(train_metrics['IoU'])
        self.train_history['precision'].append(train_metrics['Precision'])
        self.train_history['recall'].append(train_metrics['Recall'])
        self.train_history['lr'].append(lr)
        self.train_history['time'].append(epoch_time)
        
        # Val metrics
        self.val_history['epoch'].append(epoch)
        self.val_history['loss'].append(val_metrics['Loss'])
        self.val_history['mdice'].append(val_metrics['mDice'])
        self.val_history['miou'].append(val_metrics['mIoU'])
        self.val_history['precision'].append(val_metrics['Precision'])
        self.val_history['recall'].append(val_metrics['Recall'])
        
        self.epoch_times.append(epoch_time)
        
        # Save to files
        self._save_logs()
        
        # Create plots
        self._update_plots()
    
    def _save_logs(self):
        """Save logs to JSON files"""
        with open(self.train_log_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        with open(self.val_log_path, 'w') as f:
            json.dump(self.val_history, f, indent=2)
    
    def _update_plots(self):
        """Update training plots"""
        if len(self.train_history['epoch']) < 2:
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {self.experiment_name}', fontsize=16)
        
        epochs = self.train_history['epoch']
        
        # Loss plot
        axes[0, 0].plot(epochs, self.train_history['loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.val_history['loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice plot
        axes[0, 1].plot(epochs, self.train_history['dice'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.val_history['mdice'], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU plot
        axes[0, 2].plot(epochs, self.train_history['iou'], 'b-', label='Train', linewidth=2)
        axes[0, 2].plot(epochs, self.val_history['miou'], 'r-', label='Val', linewidth=2)
        axes[0, 2].set_title('IoU Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('IoU')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate plot
        axes[1, 0].plot(epochs, self.train_history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision/Recall plot
        axes[1, 1].plot(epochs, self.train_history['precision'], 'b-', label='Train Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.train_history['recall'], 'b--', label='Train Recall', linewidth=2)
        axes[1, 1].plot(epochs, self.val_history['precision'], 'r-', label='Val Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.val_history['recall'], 'r--', label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Time per epoch
        axes[1, 2].plot(epochs, self.epoch_times, 'purple', linewidth=2)
        axes[1, 2].set_title('Time per Epoch')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Time (s)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f'{self.experiment_name}_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """In tóm tắt training"""
        if not self.train_history['epoch']:
            print("Chưa có dữ liệu training!")
            return
        
        total_time = time.time() - self.start_time
        best_val_dice = max(self.val_history['mdice']) if self.val_history['mdice'] else 0
        best_epoch = self.val_history['mdice'].index(best_val_dice) + 1 if self.val_history['mdice'] else 0
        
        print("\n" + "="*60)
        print(f"TRAINING SUMMARY - {self.experiment_name}")
        print("="*60)
        print(f"Total Training Time: {total_time/3600:.2f}h")
        print(f"Total Epochs: {len(self.train_history['epoch'])}")
        print(f"Average Time/Epoch: {np.mean(self.epoch_times):.1f}s")
        print(f"Best Validation Dice: {best_val_dice:.4f} (Epoch {best_epoch})")
        
        if len(self.train_history['epoch']) > 0:
            final_train_loss = self.train_history['loss'][-1]
            final_val_loss = self.val_history['loss'][-1]
            final_train_dice = self.train_history['dice'][-1]
            final_val_dice = self.val_history['mdice'][-1]
            
            print(f"Final Train Loss: {final_train_loss:.4f}")
            print(f"Final Val Loss: {final_val_loss:.4f}")
            print(f"Final Train Dice: {final_train_dice:.4f}")
            print(f"Final Val Dice: {final_val_dice:.4f}")
        
        print(f"Logs saved to: {self.log_dir}")
        print(f"Plots saved to: {self.plots_dir}")
        print("="*60)
    
    def load_history(self, train_log_path=None, val_log_path=None):
        """Load training history từ files"""
        if train_log_path is None:
            train_log_path = self.train_log_path
        if val_log_path is None:
            val_log_path = self.val_log_path
        
        try:
            if os.path.exists(train_log_path):
                with open(train_log_path, 'r') as f:
                    self.train_history = json.load(f)
            
            if os.path.exists(val_log_path):
                with open(val_log_path, 'r') as f:
                    self.val_history = json.load(f)
                    
            print(f"Loaded history: {len(self.train_history['epoch'])} epochs")
        except Exception as e:
            print(f"Error loading history: {e}")


def visualize_training_comparison(log_paths, labels, save_path=None):
    """
    So sánh nhiều training runs
    
    Args:
        log_paths: List of (train_log_path, val_log_path) tuples
        labels: List of labels cho mỗi run
        save_path: Path để save comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Comparison', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, ((train_path, val_path), label) in enumerate(zip(log_paths, labels)):
        color = colors[i % len(colors)]
        
        try:
            # Load data
            with open(train_path, 'r') as f:
                train_data = json.load(f)
            with open(val_path, 'r') as f:
                val_data = json.load(f)
            
            epochs = train_data['epoch']
            
            # Loss comparison
            axes[0, 0].plot(epochs, val_data['loss'], 
                           color=color, label=f'{label}', linewidth=2)
            
            # Dice comparison
            axes[0, 1].plot(epochs, val_data['mdice'], 
                           color=color, label=f'{label}', linewidth=2)
            
            # IoU comparison
            axes[1, 0].plot(epochs, val_data['miou'], 
                           color=color, label=f'{label}', linewidth=2)
            
            # Learning rate comparison
            axes[1, 1].plot(epochs, train_data['lr'], 
                           color=color, label=f'{label}', linewidth=2)
            
        except Exception as e:
            print(f"Error loading {label}: {e}")
    
    # Customize plots
    axes[0, 0].set_title('Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Dice')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Validation IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show() 