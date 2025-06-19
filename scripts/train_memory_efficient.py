#!/usr/bin/env python3
"""
Training Script Memory-Efficient cho ColonFormer
Tối ưu cho GPU memory hạn chế (< 8GB)
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.colonformer import ColonFormer
from models.losses import ColonFormerLoss
from datasets import PolypDataModule
from utils.metrics import MetricTracker, evaluate_model, print_metrics


def set_seed(seed=42):
    """Set random seed cho reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MemoryEfficientTrainer:
    """
    Memory-efficient trainer với mixed precision và gradient checkpointing
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, device, save_dir, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.args = args
        
        # Mixed precision scaler
        self.scaler = GradScaler() if args.use_amp else None
        
        # Tracking variables
        self.best_dice = 0.0
        self.start_epoch = 0
        self.train_losses = []
        self.val_metrics = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Enable gradient checkpointing để save memory
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled")
    
    def train_epoch(self, epoch):
        """Train một epoch với memory optimizations"""
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(enumerate(self.train_loader), 
                   total=num_batches,
                   desc=f'Epoch {epoch}/{self.args.epochs}',
                   unit='batch')
        
        accumulated_loss = 0.0
        
        for batch_idx, batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if self.args.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss = loss / self.args.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item() * self.args.gradient_accumulation_steps
            
            # Optimizer step với gradient accumulation
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
                if self.args.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update metrics
                running_loss += accumulated_loss
                accumulated_loss = 0.0
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.args.gradient_accumulation_steps:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                    'mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB' if torch.cuda.is_available() else '0GB'
                })
            
            # Clear cache periodically để avoid memory buildup
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Scheduler step
        self.scheduler.step()
        
        avg_loss = running_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate một epoch"""
        self.model.eval()
        metric_tracker = MetricTracker()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation', leave=False):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Mixed precision forward pass
                if self.args.use_amp:
                    with autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                # Handle multiple outputs (deep supervision)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                elif isinstance(outputs, dict):
                    outputs = outputs['main']
                
                # Apply sigmoid và threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                # Update metrics
                metric_tracker.update(preds, masks)
        
        # Get metrics
        metrics = metric_tracker.get_metrics()
        self.val_metrics.append(metrics)
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'metrics': metrics,
        }
        
        # Save best model
        if metrics['dice'] > self.best_dice:
            self.best_dice = metrics['dice']
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Best model saved: {checkpoint_path} (Dice: {self.best_dice:.4f})")
        
        # Save latest model
        latest_path = os.path.join(self.save_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
    
    def train(self):
        """Main training loop"""
        print("\nBắt đầu training với memory optimizations...")
        print(f"Mixed Precision: {'Enabled' if self.args.use_amp else 'Disabled'}")
        print(f"Gradient Accumulation: {self.args.gradient_accumulation_steps} steps")
        print(f"Effective Batch Size: {self.args.batch_size * self.args.gradient_accumulation_steps}")
        print("-" * 80)
        
        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics)
            
            # Print results
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print(f"Time: {epoch_time:.1f}s")
            print(f"Train Loss: {train_loss:.4f}")
            print_metrics(val_metrics, prefix="Val")
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Max Memory Used: {memory_used:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
            
            print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='ColonFormer Memory Efficient Training')
    
    # Model parameters - default to smaller backbone
    parser.add_argument('--backbone', type=str, default='mit_b1',
                        choices=['mit_b0', 'mit_b1', 'mit_b2'],
                        help='Backbone architecture (chỉ b0, b1, b2 cho memory efficiency)')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (nhỏ để save memory)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps (để có effective batch size = 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=320,
                        help='Image size (giảm từ 352 xuống 320)')
    
    # Memory optimization
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    
    # Loss parameters
    parser.add_argument('--alpha', type=float, default=0.25,
                        help='Focal loss alpha')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--lambda_weight', type=float, default=1.0,
                        help='Loss combination weight')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='data',
                        help='Data root directory')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data workers (0 cho Windows)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Saving
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save directory')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Disable mixed precision if requested
    if args.no_amp:
        args.use_amp = False
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory Available: {gpu_memory_gb:.1f} GB")
        print(f"VRAM optimization cho GPU < 8GB enabled")
    
    # Set save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"checkpoints/colonformer_memory_efficient_{timestamp}"
    
    print(f"\nMemory Efficient Configuration:")
    print(f"  Backbone: {args.backbone} (nhỏ hơn mit_b3)")
    print(f"  Image Size: {args.img_size} (giảm từ 352)")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Mixed Precision: {'Enabled' if args.use_amp else 'Disabled'}")
    print(f"  Data Workers: {args.num_workers}")
    
    # Create data module
    print("\nPreparing data...")
    data_module = PolypDataModule(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed
    )
    
    data_module.print_statistics()
    
    # Create dataloaders
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    
    # Create model
    print(f"\nCreating model: {args.backbone}")
    model = ColonFormer(
        backbone=args.backbone,
        num_classes=args.num_classes
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Memory test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)
            if args.use_amp:
                with autocast():
                    test_output = model(dummy_input)
            else:
                test_output = model(dummy_input)
            del test_output, dummy_input
        
        model_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Model memory usage: {model_memory:.2f} GB")
        
        model.train()
        torch.cuda.empty_cache()
    
    # Create loss function
    criterion = ColonFormerLoss(
        focal_alpha=args.alpha,
        focal_gamma=args.gamma,
        loss_lambda=args.lambda_weight
    )
    
    # Create optimizer và scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"\nOptimizer: Adam (lr={args.lr})")
    print(f"Scheduler: CosineAnnealingLR (T_max={args.epochs})")
    
    # Create trainer
    trainer = MemoryEfficientTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        args=args
    )
    
    # Start training
    trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best Dice Score: {trainer.best_dice:.4f}")
    print(f"Models saved to: {args.save_dir}")


if __name__ == "__main__":
    main() 