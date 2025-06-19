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


def find_optimal_batch_size(args, device):
    """Tìm batch size tối ưu mà không bị OOM"""
    from models.colonformer import ColonFormer
    
    print("Testing different batch sizes...")
    batch_sizes = [2, 4, 6, 8, 12, 16]  # Start from 2 để avoid BatchNorm issues
    optimal_batch_size = 2  # Minimum safe batch size
    
    for batch_size in batch_sizes:
        try:
            print(f"Testing batch size: {batch_size}")
            
            # Create test model
            test_model = ColonFormer(
                backbone=args.backbone,
                num_classes=args.num_classes
            ).to(device)
            
            # Set model to training mode để test realistic conditions
            test_model.train()
            
            # Test forward và backward pass
            test_input = torch.randn(batch_size, 3, args.img_size, args.img_size).to(device)
            test_target = torch.randint(0, 2, (batch_size, 1, args.img_size, args.img_size)).float().to(device)
            
            # Forward pass
            test_output = test_model(test_input)
            if isinstance(test_output, (list, tuple)):
                test_output = test_output[0]
            
            # Dummy loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(test_output, test_target)
            
            # Backward pass
            loss.backward()
            
            # Test optimizer step
            optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-4)
            optimizer.step()
            optimizer.zero_grad()
            
            # Nếu không có lỗi, batch size này OK
            optimal_batch_size = batch_size
            print(f"Batch size {batch_size}: OK")
            
            # Clean up
            del test_model, test_input, test_target, test_output, loss, optimizer
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: OOM")
                break
            elif "Expected more than 1 value per channel" in str(e):
                print(f"Batch size {batch_size}: BatchNorm error (too small)")
                continue
            else:
                print(f"Batch size {batch_size}: Error - {str(e)}")
                continue
        except Exception as e:
            print(f"Batch size {batch_size}: Unexpected error - {str(e)}")
            continue
    
    print(f"Optimal batch size found: {optimal_batch_size}")
    return optimal_batch_size


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
        
        # Mixed precision scaler for CUDA
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
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
        """Train một epoch với gradient accumulation support"""
        self.model.train()
        metric_tracker = MetricTracker()
        
        epoch_start_time = time.time()
        
        # Gradient accumulation steps
        accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        
        # Tạo progress bar cho batches
        pbar = tqdm(enumerate(self.train_loader), 
                   total=len(self.train_loader),
                   desc=f'Epoch {epoch}/{self.args.epochs}',
                   unit='batch',
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
        
        # Initialize accumulated loss
        accumulated_loss = 0.0
        
        for batch_idx, batch in pbar:
            batch_start_time = time.time()
            
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(images)
                
                # Calculate loss với deep supervision
                if isinstance(outputs, dict):
                    # ColonFormer training mode với deep supervision
                    main_output = outputs['main']
                    coarse_output = outputs['coarse']
                    aux_outputs = outputs.get('aux', [])
                    
                    # Main loss
                    main_loss_result = self.criterion(main_output, masks)
                    if isinstance(main_loss_result, tuple):
                        main_loss = main_loss_result[0]  # Extract scalar loss
                    else:
                        main_loss = main_loss_result
                    
                    loss_total = main_loss
                    
                    # Coarse loss
                    coarse_loss_result = self.criterion(coarse_output, masks)
                    if isinstance(coarse_loss_result, tuple):
                        coarse_loss = coarse_loss_result[0]
                    else:
                        coarse_loss = coarse_loss_result
                    
                    loss_total += 0.5 * coarse_loss
                    
                    # Auxiliary losses
                    for aux_output in aux_outputs:
                        aux_loss_result = self.criterion(aux_output, masks)
                        if isinstance(aux_loss_result, tuple):
                            aux_loss = aux_loss_result[0]
                        else:
                            aux_loss = aux_loss_result
                        loss_total += 0.3 * aux_loss
                    
                    loss = loss_total
                    
                elif isinstance(outputs, (list, tuple)):
                    # Legacy format - multiple outputs
                    loss_total = 0
                    for i, output in enumerate(outputs):
                        loss_weight = 1.0 if i == 0 else 0.5  # Main output weight = 1.0, auxiliary = 0.5
                        loss_result = self.criterion(output, masks)
                        if isinstance(loss_result, tuple):
                            loss_val = loss_result[0]
                        else:
                            loss_val = loss_result
                        loss_total += loss_weight * loss_val
                    loss = loss_total
                    main_output = outputs[0]  # Main output for metrics
                else:
                    # Single output (eval mode)
                    loss_result = self.criterion(outputs, masks)
                    if isinstance(loss_result, tuple):
                        loss = loss_result[0]
                    else:
                        loss = loss_result
                    main_output = outputs
                
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
            
            accumulated_loss += loss.item()
            
            # Mixed precision backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update optimizer sau khi accumulate đủ gradients
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update metrics với accumulated loss
                metric_tracker.update(main_output, masks, accumulated_loss)
                accumulated_loss = 0.0
            
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
            postfix_dict = {
                'Loss': f'{current_metrics["Loss"]:.4f}',
                'Dice': f'{current_metrics["Dice"]:.4f}',
                'IoU': f'{current_metrics["IoU"]:.4f}',
                'LR': f'{lr:.1e}',
                'Time/batch': f'{batch_time:.2f}s',
                'CPU': f'{cpu_memory:.1f}%'
            }
            
            if accumulation_steps > 1:
                step_in_accumulation = (batch_idx % accumulation_steps) + 1
                postfix_dict['AccStep'] = f'{step_in_accumulation}/{accumulation_steps}'
            
            pbar.set_postfix(postfix_dict)
            
            # Clear cache periodically để avoid memory fragmentation
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        pbar.close()
        
        # Get epoch metrics
        epoch_metrics = metric_tracker.get_average_metrics()
        epoch_time = time.time() - epoch_start_time
        
        print(f'\nTrain Epoch {epoch} completed in {epoch_time:.1f}s')
        if accumulation_steps > 1:
            print(f'Gradient Accumulation: {accumulation_steps} steps (effective batch size: {self.args.batch_size * accumulation_steps})')
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
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(images)
                
                # Get main output nếu có deep supervision
                if isinstance(outputs, dict):
                    main_output = outputs['main']
                elif isinstance(outputs, (list, tuple)):
                    main_output = outputs[0]
                else:
                    main_output = outputs
                
                    # Calculate loss
                    loss_result = self.criterion(main_output, masks)
                    if isinstance(loss_result, tuple):
                        loss = loss_result[0]
                    else:
                        loss = loss_result
                
                # Update metrics
                metric_tracker.update(main_output, masks, loss.item())
                
                # Clear cache every 20 batches to prevent memory buildup
                if len(metric_tracker.losses) % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
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
    
    # Memory optimization parameters
    parser.add_argument('--memory_efficient', action='store_true',
                        help='Enable memory efficient training (reduce batch size and use gradient accumulation)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--max_memory_gb', type=float, default=None,
                        help='Maximum GPU memory to use (GB). Auto-adjust batch size if exceeded')
    parser.add_argument('--auto_batch_size', action='store_true',
                        help='Automatically find optimal batch size')
    
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
    
    # Memory optimization adjustments
    if args.memory_efficient:
        print("Memory efficient mode enabled!")
        if args.batch_size > 2:
            original_batch = args.batch_size
            args.batch_size = 2  # Use batch_size = 2 for BatchNorm compatibility
            args.gradient_accumulation_steps = max(original_batch // args.batch_size, 4)
            print(f"Reduced batch size from {original_batch} to {args.batch_size}")
            print(f"Set gradient accumulation steps to {args.gradient_accumulation_steps}")
        
        if args.img_size > 320:
            original_size = args.img_size
            args.img_size = 320
            print(f"Reduced image size from {original_size} to {args.img_size}")
        
        # Force minimal workers for memory efficiency
        args.num_workers = 0
        print(f"Set num_workers to {args.num_workers} for memory efficiency")
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check GPU memory và auto-adjust nếu cần
    gpu_memory_gb = 0  # Default value for CPU
    
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory Available: {gpu_memory_gb:.1f} GB")
        
        if args.max_memory_gb and gpu_memory_gb > args.max_memory_gb:
            print(f"GPU memory exceeds limit ({args.max_memory_gb} GB), enabling memory optimizations")
            args.memory_efficient = True
    else:
        print("CUDA not available, running on CPU")
        # Force memory efficient settings for CPU but maintain batch_size >= 2
        args.memory_efficient = True
        if args.batch_size < 2:
            args.batch_size = 2  # Minimum for BatchNorm
        else:
            args.batch_size = min(args.batch_size, 2)  # Cap at 2 for memory
        args.gradient_accumulation_steps = max(8 // args.batch_size, 1)
        args.num_workers = 0
    
    # Aggressive memory optimization for 8GB GPUs or CPU
    if gpu_memory_gb <= 8 or not torch.cuda.is_available():
        if torch.cuda.is_available():
            print(f"GPU <= 8GB detected ({gpu_memory_gb:.1f}GB), applying aggressive optimizations")
        else:
            print("CPU detected, applying memory optimizations")
            
        # Force minimal settings but ensure batch_size >= 2 for BatchNorm
        if args.batch_size < 2:
            args.batch_size = 2  # Minimum for BatchNorm
        else:
            args.batch_size = min(args.batch_size, 2)  # Cap at 2 for memory
            
        args.gradient_accumulation_steps = max(8 // args.batch_size, 1)  # Maintain effective batch size
        args.num_workers = 0  # Disable multiprocessing
        
        # Use smaller image size if not specified
        if args.img_size > 320:
            print(f"Reducing image size from {args.img_size} to 320 for memory")
            args.img_size = 320
        
        # Force memory efficient mode
        args.memory_efficient = True
        
        print(f"Optimized settings: batch_size={args.batch_size}, gradient_accumulation={args.gradient_accumulation_steps}")
        
    elif gpu_memory_gb < 12 and args.batch_size > 2:
        print("GPU < 12GB detected, reducing batch size to 2")
        args.batch_size = 2
        args.gradient_accumulation_steps = 4
    elif gpu_memory_gb < 16 and args.batch_size > 4:
        print("GPU < 16GB detected, reducing batch size to 4")
        args.batch_size = 4
        args.gradient_accumulation_steps = 2
    
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
    
    # Auto batch size finder
    if args.auto_batch_size:
        print("Finding optimal batch size...")
        args.batch_size = find_optimal_batch_size(args, device)
        print(f"Optimal batch size found: {args.batch_size}")
        
        # Recreate data module với batch size mới
        data_module = PolypDataModule(
            data_root=args.data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            seed=args.seed
        )
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
    
    # Memory usage check
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test forward pass để check memory - IMPORTANT: set eval mode để avoid BatchNorm error
        model.eval()  # Set to eval mode để avoid BatchNorm error với batch size = 1
        
        # Use batch size = 2 để avoid BatchNorm issues nếu model vẫn ở training mode
        test_batch_size = max(2, min(args.batch_size, 4))  # Use small but safe batch size
        dummy_input = torch.randn(test_batch_size, 3, args.img_size, args.img_size).to(device)
        
        with torch.no_grad():
            test_output = model(dummy_input)
            # Handle different return formats
            if isinstance(test_output, dict):
                print(f"Model outputs (training mode): {list(test_output.keys())}")
            else:
                print(f"Model output shape (eval mode): {test_output.shape}")
            del test_output
        
        model_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Model memory usage: {model_memory:.2f} GB (test batch size: {test_batch_size})")
        
        # Estimate total memory needed cho actual batch size
        memory_per_sample = model_memory / test_batch_size
        estimated_total = memory_per_sample * (args.batch_size + 2)  # +2 for gradients
        print(f"Estimated total memory needed: {estimated_total:.2f} GB (batch size: {args.batch_size})")
        
        if estimated_total > gpu_memory_gb * 0.9:  # 90% threshold
            print("WARNING: Estimated memory usage exceeds 90% of GPU memory!")
            print("Consider reducing batch size or image size")
            
            # Auto-suggest safer batch size
            safe_batch_size = int((gpu_memory_gb * 0.8) / memory_per_sample)
            safe_batch_size = max(1, safe_batch_size)
            print(f"SUGGESTION: Try batch size {safe_batch_size} or smaller")
        
        # Reset model back to training mode
        model.train()
        torch.cuda.empty_cache()
    
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
    
    # Memory optimization settings
    if args.gradient_accumulation_steps > 1:
        print(f"\nGradient Accumulation: {args.gradient_accumulation_steps} steps")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
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
        'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'memory_efficient': args.memory_efficient,
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