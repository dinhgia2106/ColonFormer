"""
Learning rate schedulers for ColonFormer training
"""

import math
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
    MultiStepLR,
    ExponentialLR
)


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with linear warmup
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cos_epoch = self.last_epoch - self.warmup_epochs
            cos_max_epoch = self.max_epochs - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * cos_epoch / cos_max_epoch)) / 2
                   for base_lr in self.base_lrs]


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Polynomial learning rate decay
    """
    def __init__(self, optimizer, max_epochs, power=1.0, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupPolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Polynomial decay with linear warmup
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, power=1.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.power = power
        super(WarmupPolynomialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            poly_epoch = self.last_epoch - self.warmup_epochs
            poly_max_epoch = self.max_epochs - self.warmup_epochs
            factor = (1 - poly_epoch / poly_max_epoch) ** self.power
            return [base_lr * factor for base_lr in self.base_lrs]


class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup followed by cosine annealing
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                   for base_lr in self.base_lrs]


def get_scheduler(optimizer, config, epochs=None, steps_per_epoch=None):
    """
    Get learning rate scheduler based on config
    
    Args:
        optimizer: PyTorch optimizer
        config: scheduler configuration dict
        epochs: total number of epochs
        steps_per_epoch: number of steps per epoch (for step-based schedulers)
    
    Returns:
        scheduler: PyTorch lr scheduler
    """
    scheduler_type = config.get('type', 'cosine')
    
    if scheduler_type == 'cosine':
        # Cosine annealing
        T_max = config.get('T_max', epochs)
        eta_min = config.get('eta_min', 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == 'cosine_warm_restarts':
        # Cosine annealing with warm restarts
        T_0 = config.get('T_0', 10)
        T_mult = config.get('T_mult', 2)
        eta_min = config.get('eta_min', 0)
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    
    elif scheduler_type == 'warmup_cosine':
        # Warmup + cosine annealing
        warmup_epochs = config.get('warmup_epochs', 5)
        eta_min = config.get('eta_min', 0)
        return WarmupCosineAnnealingLR(optimizer, warmup_epochs, epochs, eta_min)
    
    elif scheduler_type == 'linear_warmup_cosine':
        # Linear warmup + cosine annealing
        warmup_epochs = config.get('warmup_epochs', 5)
        eta_min = config.get('eta_min', 0)
        return LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs, epochs, eta_min)
    
    elif scheduler_type == 'polynomial':
        # Polynomial decay
        power = config.get('power', 1.0)
        return PolynomialLR(optimizer, epochs, power)
    
    elif scheduler_type == 'warmup_polynomial':
        # Warmup + polynomial decay
        warmup_epochs = config.get('warmup_epochs', 5)
        power = config.get('power', 1.0)
        return WarmupPolynomialLR(optimizer, warmup_epochs, epochs, power)
    
    elif scheduler_type == 'plateau':
        # Reduce on plateau
        mode = config.get('mode', 'max')
        factor = config.get('factor', 0.5)
        patience = config.get('patience', 10)
        threshold = config.get('threshold', 1e-4)
        min_lr = config.get('min_lr', 0)
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, 
                               patience=patience, threshold=threshold, min_lr=min_lr)
    
    elif scheduler_type == 'step':
        # Step decay
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'multistep':
        # Multi-step decay
        milestones = config.get('milestones', [30, 60, 90])
        gamma = config.get('gamma', 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        # Exponential decay
        gamma = config.get('gamma', 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'none' or scheduler_type is None:
        # No scheduler
        return None
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_warmup_scheduler(optimizer, warmup_steps, warmup_factor=0.1):
    """
    Get a warmup scheduler for the first few steps/epochs
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: number of warmup steps
        warmup_factor: initial learning rate factor
    
    Returns:
        lambda function for warmup
    """
    def warmup_lambda(step):
        if step < warmup_steps:
            return warmup_factor + (1 - warmup_factor) * step / warmup_steps
        else:
            return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)


class SchedulerManager:
    """
    Manager for handling multiple schedulers (e.g., warmup + main scheduler)
    """
    def __init__(self, optimizer, main_config, warmup_config=None, epochs=None, steps_per_epoch=None):
        self.optimizer = optimizer
        self.main_scheduler = get_scheduler(optimizer, main_config, epochs, steps_per_epoch)
        self.warmup_scheduler = None
        self.warmup_epochs = 0
        
        if warmup_config:
            self.warmup_epochs = warmup_config.get('epochs', 0)
            if self.warmup_epochs > 0:
                warmup_steps = self.warmup_epochs * (steps_per_epoch or 1)
                warmup_factor = warmup_config.get('factor', 0.1)
                self.warmup_scheduler = get_warmup_scheduler(optimizer, warmup_steps, warmup_factor)
        
        self.current_epoch = 0
        self.current_step = 0
    
    def step(self, epoch=None, metrics=None):
        """Step the scheduler"""
        if epoch is not None:
            self.current_epoch = epoch
        
        # Handle warmup
        if self.warmup_scheduler and self.current_epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        elif self.main_scheduler:
            # Step main scheduler
            if isinstance(self.main_scheduler, ReduceLROnPlateau):
                if metrics is not None:
                    self.main_scheduler.step(metrics)
            else:
                self.main_scheduler.step()
    
    def step_batch(self):
        """Step for batch-based schedulers"""
        self.current_step += 1
        
        # For step-based warmup
        if self.warmup_scheduler and self.current_step < self.warmup_epochs * self.steps_per_epoch:
            self.warmup_scheduler.step()
    
    def get_last_lr(self):
        """Get last learning rate"""
        if self.warmup_scheduler and self.current_epoch < self.warmup_epochs:
            return self.warmup_scheduler.get_last_lr()
        elif self.main_scheduler:
            return self.main_scheduler.get_last_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Get state dict"""
        state = {
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'main_scheduler': self.main_scheduler.state_dict() if self.main_scheduler else None,
            'warmup_scheduler': self.warmup_scheduler.state_dict() if self.warmup_scheduler else None
        }
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.current_epoch = state_dict['current_epoch']
        self.current_step = state_dict['current_step']
        
        if self.main_scheduler and state_dict['main_scheduler']:
            self.main_scheduler.load_state_dict(state_dict['main_scheduler'])
        
        if self.warmup_scheduler and state_dict['warmup_scheduler']:
            self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler']) 