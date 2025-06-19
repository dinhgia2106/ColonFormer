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
        
        # Loss function config
        if criterion is not None:
            loss_config = {
                'type': criterion.__class__.__name__,
            }
            
            # Get loss-specific parameters
            if hasattr(criterion, 'focal_loss'):
                loss_config['focal_alpha'] = criterion.focal_loss.alpha
                loss_config['focal_gamma'] = criterion.focal_loss.gamma
            if hasattr(criterion, 'loss_lambda'):
                loss_config['loss_lambda'] = criterion.loss_lambda
            if hasattr(criterion, 'deep_supervision'):
                loss_config['deep_supervision'] = criterion.deep_supervision
                
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
        """Print experiment summary"""
        print("\n" + "="*70)
        print(f"EXPERIMENT SUMMARY - {self.experiment_id}")
        print("="*70)
        print(f"Name: {self.experiment_name}")
        print(f"Directory: {self.exp_dir}")
        
        if 'model' in self.config:
            model_info = self.config['model']
            print(f"\nModel: {model_info.get('name', 'Unknown')}")
            print(f"Backbone: {model_info.get('backbone', 'Unknown')}")
            print(f"Parameters: {model_info.get('total_parameters', 0):,}")
            print(f"Size: {model_info.get('model_size_mb', 0):.1f} MB")
        
        if 'training' in self.config:
            train_info = self.config['training']
            print(f"\nTraining:")
            print(f"Epochs: {train_info.get('epochs', 'Unknown')}")
            print(f"Batch Size: {train_info.get('batch_size', 'Unknown')}")
            print(f"Learning Rate: {train_info.get('lr', 'Unknown')}")
            if 'optimizer' in train_info:
                print(f"Optimizer: {train_info['optimizer'].get('type', 'Unknown')}")
        
        if 'final' in self.results:
            final_results = self.results['final']
            print(f"\nResults:")
            print(f"Best Epoch: {final_results.get('best_epoch', 'Unknown')}")
            print(f"Best Val Dice: {final_results.get('best_val_metrics', {}).get('mDice', 0):.4f}")
            print(f"Final Val Dice: {final_results.get('final_val_metrics', {}).get('mDice', 0):.4f}")
        
        print("="*70)


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