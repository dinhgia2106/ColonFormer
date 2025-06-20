"""
Experiment Tracker cho ColonFormer
Log toàn bộ tham số, model config, và tạo unique experiment ID
"""

import os
import json
import uuid
import time
import socket
import torch
import numpy as np
from datetime import datetime
from collections import OrderedDict


class ExperimentTracker:
    """
    Track toàn bộ experiment với unique ID và comprehensive logging
    """
    
    def __init__(self, save_dir, experiment_name=None):
        self.save_dir = save_dir
        self.experiment_id = str(uuid.uuid4())[:8]  # Short unique ID
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if experiment_name:
            self.experiment_name = f"{experiment_name}_{self.experiment_id}"
        else:
            self.experiment_name = f"exp_{self.timestamp}_{self.experiment_id}"
        
        # Create experiment directory
        self.exp_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # File paths
        self.config_file = os.path.join(self.exp_dir, 'experiment_config.json')
        self.results_file = os.path.join(self.exp_dir, 'results.json')
        self.log_file = os.path.join(self.exp_dir, 'training.log')
        
        # Initialize experiment info
        self.config = OrderedDict()
        self.results = OrderedDict()
        self.training_history = []
        
        # System info
        self._log_system_info()
        
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Experiment Directory: {self.exp_dir}")
    
    def _log_system_info(self):
        """Log system information"""
        self.config['system'] = {
            'hostname': socket.gethostname(),
            'python_version': f"{torch.version.__version__}",
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'timestamp': self.timestamp,
            'experiment_id': self.experiment_id
        }
        
        if torch.cuda.is_available():
            self.config['system']['gpu_names'] = [
                torch.cuda.get_device_name(i) 
                for i in range(torch.cuda.device_count())
            ]
    
    def log_model_config(self, model, model_name=None):
        """Log model configuration và architecture"""
        if model_name is None:
            model_name = model.__class__.__name__
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_config = {
            'name': model_name,
            'class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assume float32
        }
        
        # Try to get model-specific config
        if hasattr(model, 'backbone_name'):
            model_config['backbone'] = model.backbone_name
        if hasattr(model, 'num_classes'):
            model_config['num_classes'] = model.num_classes
        if hasattr(model, 'img_size'):
            model_config['img_size'] = model.img_size
        if hasattr(model, 'deep_supervision'):
            model_config['deep_supervision'] = model.deep_supervision
        if hasattr(model, 'use_refinement'):
            model_config['use_refinement'] = model.use_refinement
        
        # Get model string representation
        model_config['architecture'] = str(model)
        
        self.config['model'] = model_config
        
        return model_config
    
    def log_training_config(self, args=None, optimizer=None, scheduler=None, 
                          criterion=None, **kwargs):
        """Log training configuration"""
        training_config = {}
        
        # Training arguments
        if args is not None:
            if hasattr(args, '__dict__'):
                training_config.update(vars(args))
            elif isinstance(args, dict):
                training_config.update(args)
        
        # Additional kwargs
        training_config.update(kwargs)
        
        # Optimizer config
        if optimizer is not None:
            opt_config = {
                'type': optimizer.__class__.__name__,
                'lr': optimizer.param_groups[0]['lr'],
                'params': {}
            }
            
            # Get optimizer-specific parameters
            for key, value in optimizer.param_groups[0].items():
                if key not in ['params', 'lr']:
                    opt_config['params'][key] = value
            
            training_config['optimizer'] = opt_config
        
        # Scheduler config
        if scheduler is not None:
            scheduler_config = {
                'type': scheduler.__class__.__name__,
            }
            
            # Get scheduler-specific parameters
            if hasattr(scheduler, 'T_max'):
                scheduler_config['T_max'] = scheduler.T_max
            if hasattr(scheduler, 'eta_min'):
                scheduler_config['eta_min'] = scheduler.eta_min
            if hasattr(scheduler, 'step_size'):
                scheduler_config['step_size'] = scheduler.step_size
            if hasattr(scheduler, 'gamma'):
                scheduler_config['gamma'] = scheduler.gamma
                
            training_config['scheduler'] = scheduler_config
        
        # Loss function config - Chi tiết đầy đủ
        if criterion is not None:
            loss_config = {
                'type': criterion.__class__.__name__,
                'description': 'ColonFormer Loss Function'
            }
            
            # Get loss-specific parameters từ ColonFormerLoss
            if hasattr(criterion, 'focal_loss'):
                loss_config['focal_alpha'] = criterion.focal_loss.alpha
                loss_config['focal_gamma'] = criterion.focal_loss.gamma
                loss_config['focal_description'] = f"Weighted Focal Loss với α={criterion.focal_loss.alpha}, γ={criterion.focal_loss.gamma}"
            
            if hasattr(criterion, 'iou_loss'):
                loss_config['iou_description'] = "Weighted IoU Loss với distance-based weighting"
                
            if hasattr(criterion, 'loss_lambda'):
                loss_config['loss_lambda'] = criterion.loss_lambda
                loss_config['formula'] = f"L_total = {criterion.loss_lambda} * L_wfocal + L_wiou"
            
            if hasattr(criterion, 'deep_supervision'):
                loss_config['deep_supervision'] = criterion.deep_supervision
            else:
                loss_config['deep_supervision'] = True  # Default cho ColonFormer
                
            # Thêm thông tin về distance weighting
            loss_config['distance_weighting'] = True
            loss_config['distance_weight_description'] = "Border proximity weighting cho IoU loss"
            
            # Thêm các thông số quan trọng khác
            loss_config['components'] = [
                "Weighted Focal Loss (classification)",
                "Weighted IoU Loss (segmentation)",
                "Distance-based border weighting",
                "Deep supervision support"
            ]
            
            loss_config['optimization_impact'] = [
                "focal_alpha: Controls class imbalance weighting",
                "focal_gamma: Controls hard example focusing", 
                "loss_lambda: Balances focal vs IoU contribution",
                "distance_weighting: Emphasizes polyp boundaries"
            ]
                
            training_config['criterion'] = loss_config
        
        self.config['training'] = training_config
        
        return training_config
    
    def log_data_config(self, data_module=None, **kwargs):
        """Log data configuration"""
        data_config = {}
        
        if data_module is not None:
            if hasattr(data_module, 'data_root'):
                data_config['data_root'] = data_module.data_root
            if hasattr(data_module, 'img_size'):
                data_config['img_size'] = data_module.img_size
            if hasattr(data_module, 'batch_size'):
                data_config['batch_size'] = data_module.batch_size
            if hasattr(data_module, 'num_workers'):
                data_config['num_workers'] = data_module.num_workers
            if hasattr(data_module, 'val_split'):
                data_config['val_split'] = data_module.val_split
            if hasattr(data_module, 'train_size'):
                data_config['train_size'] = data_module.train_size
            if hasattr(data_module, 'val_size'):
                data_config['val_size'] = data_module.val_size
        
        data_config.update(kwargs)
        self.config['data'] = data_config
        
        return data_config
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def log_epoch_results(self, epoch, train_metrics, val_metrics, lr=None, epoch_time=None):
        """Log results của một epoch"""
        epoch_result = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        
        if lr is not None:
            epoch_result['learning_rate'] = lr
        if epoch_time is not None:
            epoch_result['epoch_time'] = epoch_time
        
        self.training_history.append(epoch_result)
        
        # Update current results
        self.results['current_epoch'] = epoch
        self.results['best_val_dice'] = max(
            [h['val_metrics'].get('mDice', 0) for h in self.training_history]
        )
        self.results['training_history'] = self.training_history
        
        # Save to file
        self._save_results()
    
    def log_final_results(self, test_results=None, best_checkpoint_path=None):
        """Log final results của experiment"""
        if len(self.training_history) > 0:
            best_epoch_idx = np.argmax([
                h['val_metrics'].get('mDice', 0) for h in self.training_history
            ])
            best_epoch_data = self.training_history[best_epoch_idx]
            
            self.results['final'] = {
                'total_epochs': len(self.training_history),
                'best_epoch': best_epoch_data['epoch'],
                'best_val_metrics': best_epoch_data['val_metrics'],
                'final_train_metrics': self.training_history[-1]['train_metrics'],
                'final_val_metrics': self.training_history[-1]['val_metrics'],
            }
        
        if test_results is not None:
            self.results['test_results'] = test_results
        
        if best_checkpoint_path is not None:
            self.results['best_checkpoint'] = best_checkpoint_path
        
        self.results['experiment_completed'] = datetime.now().isoformat()
        
        # Save final results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save results to file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def get_summary(self):
        """Get experiment summary"""
        summary = {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'directory': self.exp_dir,
            'config': self.config,
            'results': self.results,
            'status': 'completed' if 'experiment_completed' in self.results else 'running'
        }
        
        return summary
    
    def print_summary(self):
        """In summary của experiment"""
        print("\n" + "="*80)
        print(f"EXPERIMENT SUMMARY - {self.experiment_name}")
        print("="*80)
        
        # System info
        if 'system' in self.config:
            sys_info = self.config['system']
            print(f"System: {sys_info.get('hostname', 'Unknown')}")
            print(f"PyTorch: {sys_info.get('pytorch_version', 'Unknown')}")
            if sys_info.get('cuda_available', False):
                print(f"CUDA: {sys_info.get('cuda_version', 'Unknown')}")
                gpu_names = sys_info.get('gpu_names', [])
                if gpu_names:
                    print(f"GPU: {', '.join(gpu_names)}")
        
        # Model info
        if 'model' in self.config:
            model_info = self.config['model']
            print(f"\nModel: {model_info.get('name', 'Unknown')}")
            print(f"Parameters: {model_info.get('total_parameters', 0):,} total, {model_info.get('trainable_parameters', 0):,} trainable")
            print(f"Model Size: {model_info.get('model_size_mb', 0):.1f} MB")
            if 'backbone' in model_info:
                print(f"Backbone: {model_info['backbone']}")
        
        # Loss Configuration - Chi tiết đầy đủ 
        if 'training' in self.config and 'criterion' in self.config['training']:
            criterion_info = self.config['training']['criterion']
            print(f"\nLoss Configuration:")
            print(f"  Type: {criterion_info.get('type', 'Unknown')}")
            print(f"  Description: {criterion_info.get('description', 'N/A')}")
            
            if 'formula' in criterion_info:
                print(f"  Formula: {criterion_info['formula']}")
            
            if 'focal_alpha' in criterion_info:
                print(f"  Focal Alpha (α): {criterion_info['focal_alpha']}")
            if 'focal_gamma' in criterion_info:
                print(f"  Focal Gamma (γ): {criterion_info['focal_gamma']}")
            if 'loss_lambda' in criterion_info:
                print(f"  Lambda Weight (λ): {criterion_info['loss_lambda']}")
            
            print(f"  Deep Supervision: {criterion_info.get('deep_supervision', 'Unknown')}")
            print(f"  Distance Weighting: {criterion_info.get('distance_weighting', 'Unknown')}")
            
            if 'components' in criterion_info:
                print(f"  Components:")
                for comp in criterion_info['components']:
                    print(f"    - {comp}")
            
            if 'optimization_impact' in criterion_info:
                print(f"  Optimization Impact:")
                for impact in criterion_info['optimization_impact']:
                    print(f"    - {impact}")
        
        # Training info
        if 'training' in self.config:
            train_info = self.config['training']
            print(f"\nTraining Configuration:")
            print(f"  Epochs: {train_info.get('epochs', 'Unknown')}")
            print(f"  Batch Size: {train_info.get('batch_size', 'Unknown')}")
            print(f"  Learning Rate: {train_info.get('learning_rate', 'Unknown')}")
            if 'optimizer' in train_info:
                opt_info = train_info['optimizer']
                print(f"  Optimizer: {opt_info.get('type', 'Unknown')} (lr={opt_info.get('lr', 'Unknown')})")
            if 'scheduler' in train_info:
                sch_info = train_info['scheduler']
                print(f"  Scheduler: {sch_info.get('type', 'Unknown')}")
                if 'T_max' in sch_info:
                    print(f"    T_max: {sch_info['T_max']}")
        
        # Data info
        if 'data' in self.config:
            data_info = self.config['data']
            print(f"\nData Configuration:")
            print(f"  Data Root: {data_info.get('data_root', 'Unknown')}")
            print(f"  Image Size: {data_info.get('img_size', 'Unknown')}")
            print(f"  Train Size: {data_info.get('train_size', 'Unknown')}")
            print(f"  Val Size: {data_info.get('val_size', 'Unknown')}")
            print(f"  Val Split: {data_info.get('val_split', 'Unknown')}")
        
        # Results summary
        if self.training_history:
            print(f"\nTraining Results:")
            best_epoch = max(self.training_history, key=lambda x: x.get('val_metrics', {}).get('mDice', 0))
            print(f"  Best Epoch: {best_epoch.get('epoch', 'Unknown')}")
            print(f"  Best Val Dice: {best_epoch.get('val_metrics', {}).get('mDice', 0):.4f}")
            print(f"  Best Val IoU: {best_epoch.get('val_metrics', {}).get('mIoU', 0):.4f}")
            print(f"  Final Train Loss: {self.training_history[-1].get('train_metrics', {}).get('Loss', 0):.4f}")
        
        print(f"\nExperiment Directory: {self.exp_dir}")
        print(f"Config File: {self.config_file}")
        print("="*80)
    
    def print_loss_comparison(self, other_experiments=None):
        """In bảng so sánh loss configuration với các experiments khác"""
        print("\n" + "="*100)
        print("LOSS CONFIGURATION COMPARISON")
        print("="*100)
        
        # Current experiment
        if 'training' in self.config and 'criterion' in self.config['training']:
            criterion_info = self.config['training']['criterion']
            print(f"\nCurrent Experiment: {self.experiment_name}")
            print(f"  Loss Type: {criterion_info.get('type', 'Unknown')}")
            print(f"  Formula: {criterion_info.get('formula', 'N/A')}")
            print(f"  Focal Alpha: {criterion_info.get('focal_alpha', 'N/A')}")
            print(f"  Focal Gamma: {criterion_info.get('focal_gamma', 'N/A')}")
            print(f"  Lambda Weight: {criterion_info.get('loss_lambda', 'N/A')}")
            print(f"  Deep Supervision: {criterion_info.get('deep_supervision', 'N/A')}")
            print(f"  Distance Weighting: {criterion_info.get('distance_weighting', 'N/A')}")
        
        # Other experiments (if provided)
        if other_experiments:
            for i, exp in enumerate(other_experiments):
                print(f"\nComparison Experiment {i+1}: {exp.get('name', 'Unknown')}")
                criterion_info = exp.get('criterion', {})
                print(f"  Loss Type: {criterion_info.get('type', 'Unknown')}")
                print(f"  Formula: {criterion_info.get('formula', 'N/A')}")
                print(f"  Focal Alpha: {criterion_info.get('focal_alpha', 'N/A')}")
                print(f"  Focal Gamma: {criterion_info.get('focal_gamma', 'N/A')}")
                print(f"  Lambda Weight: {criterion_info.get('loss_lambda', 'N/A')}")
                print(f"  Performance: {exp.get('best_dice', 'N/A')}")
        
        print("="*100)


def load_experiment(experiment_dir):
    """Load experiment từ directory"""
    config_file = os.path.join(experiment_dir, 'experiment_config.json')
    results_file = os.path.join(experiment_dir, 'results.json')
    
    experiment_data = {}
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            experiment_data['config'] = json.load(f)
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            experiment_data['results'] = json.load(f)
    
    # Extract experiment info
    experiment_data['experiment_id'] = experiment_data.get('config', {}).get('system', {}).get('experiment_id', 'unknown')
    experiment_data['experiment_name'] = os.path.basename(experiment_dir)
    experiment_data['directory'] = experiment_dir
    
    return experiment_data


def list_experiments(base_dir):
    """List tất cả experiments trong base directory"""
    experiments = []
    
    if not os.path.exists(base_dir):
        return experiments
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            config_file = os.path.join(item_path, 'experiment_config.json')
            if os.path.exists(config_file):
                try:
                    exp_data = load_experiment(item_path)
                    experiments.append(exp_data)
                except Exception as e:
                    print(f"Error loading experiment {item}: {e}")
    
    # Sort by timestamp
    experiments.sort(key=lambda x: x.get('config', {}).get('system', {}).get('timestamp', ''), reverse=True)
    
    return experiments 