"""
Training Script cho ColonFormer
Theo plan: Adam optimizer, lr=1e-4, Cosine Annealing scheduler, batch size=8, epochs=20
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
import psutil

from models.colonformer import ColonFormer
from models.losses import ColonFormerLoss
from datasets import PolypDataModule
from utils.metrics import MetricTracker, evaluate_model, print_metrics
from utils.logger import TrainingLogger
from utils.experiment_tracker import ExperimentTracker


def set_seed(seed=42):
    """Set random seed cho reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, best_dice, 
                   save_dir, filename=None):
    """Lưu model checkpoint"""
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_dice': best_dice,
    }
    
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_dice = checkpoint['best_dice']
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, best dice: {best_dice:.4f}")
    
    return epoch, best_dice


class Trainer:
    """
    Trainer class cho ColonFormer
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
        
        # Tracking variables
        self.best_dice = 0.0
        self.start_epoch = 0
        self.train_losses = []
        self.val_metrics = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize experiment tracker và logger
        experiment_name = f"colonformer_{args.backbone}"
        self.experiment_tracker = ExperimentTracker(save_dir, experiment_name)
        self.logger = TrainingLogger(self.experiment_tracker.exp_dir, 
                                   self.experiment_tracker.experiment_name)
        
        # Resume from checkpoint if specified
        if args.resume:
            self.start_epoch, self.best_dice = load_checkpoint(
                model, optimizer, scheduler, args.resume
            )
    
    def train_epoch(self, epoch):
        """Train một epoch"""
        self.model.train()
        metric_tracker = MetricTracker()
        
        epoch_start_time = time.time()
        
        # Tạo progress bar cho batches
        pbar = tqdm(enumerate(self.train_loader), 
                   total=len(self.train_loader),
                   desc=f'Epoch {epoch}/{self.args.epochs}',
                   unit='batch',
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
        
        for batch_idx, batch in pbar:
            batch_start_time = time.time()
            
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss với deep supervision
            if isinstance(outputs, (list, tuple)):
                # Deep supervision - multiple outputs
                loss_total = 0
                for i, output in enumerate(outputs):
                    loss_weight = 1.0 if i == 0 else 0.5  # Main output weight = 1.0, auxiliary = 0.5
                    loss_total += loss_weight * self.criterion(output, masks)
                loss = loss_total
                main_output = outputs[0]  # Main output for metrics
            else:
                loss = self.criterion(outputs, masks)
                main_output = outputs
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metric_tracker.update(main_output, masks, loss.item())
            
            # Get current metrics
            current_metrics = metric_tracker.get_current_metrics()
            lr = self.optimizer.param_groups[0]['lr']
            
            # Calculate batch time and memory usage
            batch_time = time.time() - batch_start_time
            
            # GPU memory usage (nếu có GPU)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                gpu_memory_str = f', GPU: {gpu_memory:.1f}GB'
            else:
                gpu_memory_str = ''
            
            # CPU memory usage  
            cpu_memory = psutil.virtual_memory().percent
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{current_metrics["Loss"]:.4f}',
                'Dice': f'{current_metrics["Dice"]:.4f}',
                'IoU': f'{current_metrics["IoU"]:.4f}',
                'LR': f'{lr:.1e}',
                'Time/batch': f'{batch_time:.2f}s',
                'CPU': f'{cpu_memory:.1f}%'
            })
        
        pbar.close()
        
        # Get epoch metrics
        epoch_metrics = metric_tracker.get_average_metrics()
        epoch_time = time.time() - epoch_start_time
        
        print(f'\nTrain Epoch {epoch} completed in {epoch_time:.1f}s')
        print_metrics(epoch_metrics, "Train")
        
        self.train_losses.append(epoch_metrics['Loss'])
        
        return epoch_metrics
    
    def validate_epoch(self, epoch):
        """Validate một epoch"""
        print(f'\nValidating epoch {epoch}...')
        
        self.model.eval()
        metric_tracker = MetricTracker()
        
        with torch.no_grad():
            # Progress bar cho validation
            pbar = tqdm(self.val_loader, 
                       desc='Validation',
                       unit='batch',
                       bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Get main output nếu có deep supervision
                if isinstance(outputs, (list, tuple)):
                    main_output = outputs[0]
                else:
                    main_output = outputs
                
                # Calculate loss
                loss = self.criterion(main_output, masks)
                
                # Update metrics
                metric_tracker.update(main_output, masks, loss.item())
                
                # Update progress bar
                current_metrics = metric_tracker.get_current_metrics()
                pbar.set_postfix({
                    'Loss': f'{current_metrics["Loss"]:.4f}',
                    'Dice': f'{current_metrics["Dice"]:.4f}',
                    'IoU': f'{current_metrics["IoU"]:.4f}'
                })
            
            pbar.close()
        
        # Get validation metrics
        val_metrics = metric_tracker.get_average_metrics()
        print_metrics(val_metrics, "Val")
        
        self.val_metrics.append(val_metrics)
        
        return val_metrics
    
    def train(self):
        """Main training loop"""
        print(f"Bắt đầu training từ epoch {self.start_epoch + 1} đến {self.args.epochs}")
        print(f"Device: {self.device}")
        
        # In chi tiết về cấu hình mô hình và loss
        print("\n" + "="*60)
        print("CẤU HÌNH MÔ HÌNH VÀ TRAINING")
        print("="*60)
        
        # Thông tin mô hình
        print(f"Model: ColonFormer-{self.args.backbone}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Thông tin Loss Function
        print(f"\nLoss Configuration:")
        print(f"  Loss Type: ColonFormerLoss (λ * L_wfocal + L_wiou)")
        print(f"  Focal Loss Alpha (α): {self.args.alpha}")
        print(f"  Focal Loss Gamma (γ): {self.args.gamma}")
        print(f"  Loss Lambda (λ): {self.args.lambda_weight}")
        print(f"  Distance Weight: Enabled (border proximity weighting)")
        print(f"  Deep Supervision: Enabled (main + auxiliary outputs)")
        
        # Thông tin Training
        print(f"\nTraining Configuration:")
        print(f"  Optimizer: Adam")
        print(f"  Learning Rate: {self.args.lr}")
        print(f"  Scheduler: CosineAnnealingLR (T_max={self.args.epochs})")
        print(f"  Batch Size: {self.args.batch_size}")
        print(f"  Epochs: {self.args.epochs}")
        print(f"  Image Size: {self.args.img_size}x{self.args.img_size}")
        print(f"  Random Seed: {self.args.seed}")
        
        # Thông tin Data
        print(f"\nData Configuration:")
        print(f"  Data Root: {self.args.data_root}")
        print(f"  Validation Split: {self.args.val_split}")
        print(f"  Num Workers: {self.args.num_workers}")
        print(f"  Augmentation: Enabled (flip, rotate, scale, brightness)")
        
        # Thông tin Device và Memory
        print(f"\nDevice Information:")
        print(f"  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
        
        print(f"  Save Directory: {self.save_dir}")
        print("="*60)
        
        # Progress bar cho toàn bộ training
        epoch_range = range(self.start_epoch + 1, self.args.epochs + 1)
        training_start_time = time.time()
        
        for epoch in epoch_range:
            epoch_start_time = time.time()
            print(f'\nEpoch {epoch}/{self.args.epochs}')
            print('-' * 50)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Calculate times
            epoch_time = time.time() - epoch_start_time
            total_elapsed = time.time() - training_start_time
            epochs_done = epoch - self.start_epoch
            epochs_remaining = self.args.epochs - epoch
            if epochs_done > 0:
                avg_epoch_time = total_elapsed / epochs_done
                eta = avg_epoch_time * epochs_remaining
                eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m"
            else:
                eta_str = "N/A"
            
            # Save best model
            current_dice = val_metrics['mDice']
            is_best = current_dice > self.best_dice
            if is_best:
                self.best_dice = current_dice
                best_checkpoint = save_checkpoint(
                    self.model, self.optimizer, self.scheduler, 
                    epoch, self.best_dice, self.save_dir, 'best_model.pth'
                )
                print(f"\n*** NEW BEST MODEL! Dice: {self.best_dice:.4f} ***")
            
            # Save periodic checkpoint
            if epoch % self.args.save_interval == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.best_dice, self.save_dir,
                    f'checkpoint_epoch_{epoch}.pth'
                )
            
            # Log to experiment tracker và training logger
            self.experiment_tracker.log_epoch_results(epoch, train_metrics, val_metrics, new_lr, epoch_time)
            self.logger.log_epoch(epoch, train_metrics, val_metrics, new_lr, epoch_time)
            
            # Summary với thông tin chi tiết
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Time: {epoch_time:.1f}s | ETA: {eta_str} | LR: {old_lr:.1e} -> {new_lr:.1e}")
            print(f"  Train - Loss: {train_metrics['Loss']:.4f}, Dice: {train_metrics['Dice']:.4f}, IoU: {train_metrics['IoU']:.4f}")
            print(f"  Val   - Loss: {val_metrics['Loss']:.4f}, Dice: {val_metrics['mDice']:.4f}, IoU: {val_metrics['mIoU']:.4f}")
            print(f"  Best Dice: {self.best_dice:.4f} {'[NEW]' if is_best else ''}")
            
            # Thông tin bộ nhớ nếu có GPU
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.max_memory_allocated() / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"  GPU Memory: {gpu_memory_used:.1f}/{gpu_memory_total:.1f} GB ({gpu_memory_used/gpu_memory_total*100:.1f}%)")
                torch.cuda.reset_peak_memory_stats()
            
            print("="*50)
        
        total_time = time.time() - training_start_time
        print(f"\nTraining completed in {total_time/3600:.1f}h!")
        print(f"Best Dice achieved: {self.best_dice:.4f}")
        print(f"Best model saved at: {os.path.join(self.save_dir, 'best_model.pth')}")
        
        # Log final results và print summary
        self.experiment_tracker.log_final_results(best_checkpoint_path=os.path.join(self.save_dir, 'best_model.pth'))
        self.experiment_tracker.print_summary()
        self.logger.print_summary()


def main():
    parser = argparse.ArgumentParser(description='ColonFormer Training')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='mit_b3',
                        choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'],
                        help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes')
    
    # Training parameters theo plan
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=352,
                        help='Image size')
    
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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data workers')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save interval')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"checkpoints/colonformer_{args.backbone}_{timestamp}"
    
    # Create data module
    print("Preparing data...")
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
    print(f"Creating model: {args.backbone}")
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
    
    # Create loss function
    criterion = ColonFormerLoss(
        focal_alpha=args.alpha,
        focal_gamma=args.gamma,
        loss_lambda=args.lambda_weight
    )
    
    # In thông tin chi tiết về loss function
    print(f"\nCreating loss function:")
    print(f"  Type: ColonFormerLoss")
    print(f"  Formula: λ * L_wfocal + L_wiou")
    print(f"  Focal Alpha (α): {args.alpha}")
    print(f"  Focal Gamma (γ): {args.gamma}")
    print(f"  Lambda Weight (λ): {args.lambda_weight}")
    print(f"  Distance Weighting: Enabled")
    print(f"  Deep Supervision: Enabled")
    
    # Create optimizer và scheduler theo plan
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"\nCreating optimizer và scheduler:")
    print(f"  Optimizer: Adam (lr={args.lr})")
    print(f"  Scheduler: CosineAnnealingLR (T_max={args.epochs})")
    
    # Create trainer
    trainer = Trainer(
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
    
    # Log all configurations với thông tin chi tiết
    print("\nLogging experiment configurations...")
    trainer.experiment_tracker.log_model_config(model, f"ColonFormer-{args.backbone}")
    
    # Log thông tin training config với loss details
    training_config = {
        'model_name': f"ColonFormer-{args.backbone}",
        'backbone': args.backbone,
        'num_classes': args.num_classes,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'optimizer': 'Adam',
        'scheduler': 'CosineAnnealingLR',
        'scheduler_T_max': args.epochs,
        'loss_type': 'ColonFormerLoss',
        'loss_formula': 'λ * L_wfocal + L_wiou',
        'focal_alpha': args.alpha,
        'focal_gamma': args.gamma,
        'loss_lambda': args.lambda_weight,
        'distance_weighting': True,
        'deep_supervision': True,
        'seed': args.seed,
        'device': str(device),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'save_interval': args.save_interval,
        'log_interval': args.log_interval
    }
    
    trainer.experiment_tracker.config.update(training_config)
    trainer.experiment_tracker.log_training_config(args, optimizer, scheduler, criterion)
    trainer.experiment_tracker.log_data_config(data_module)
    trainer.experiment_tracker.save_config()
    
    print("Configuration logging completed!")
    print(f"All parameters will be saved to: {trainer.experiment_tracker.exp_dir}")
    print(f"Loss configuration saved for future reference and comparison!")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 