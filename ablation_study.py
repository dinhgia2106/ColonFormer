"""
Ablation Study cho ColonFormer
Theo Bảng 5 trong paper:
1. Baseline: MiT-B3 + UPER Decoder only
2. + CFP: Thêm CFP blocks  
3. + RA-RA: Thêm RA-RA blocks (full ColonFormer)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from datetime import datetime

from models.colonformer import ColonFormer
from models.losses import ColonFormerLoss
from datasets import PolypDataModule
from utils.metrics import evaluate_model, print_metrics
from train import Trainer, set_seed


class AblationColonFormer(nn.Module):
    """
    ColonFormer với khả năng enable/disable các components để ablation study
    """
    def __init__(self, backbone='mit_b3', num_classes=1, enable_cfp=True, enable_rara=True):
        super(AblationColonFormer, self).__init__()
        
        # Import components
        from models.backbones import MiTBackbone
        from models.components import UPerDecoder, CFPBlock, RARABlock
        
        self.enable_cfp = enable_cfp
        self.enable_rara = enable_rara
        
        # Encoder
        self.encoder = MiTBackbone(backbone)
        
        # Decoder  
        embed_dims = self.encoder.embed_dims
        self.decoder = UPerDecoder(
            in_channels=embed_dims,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False
        )
        
        # Refinement modules (optional)
        if self.enable_cfp:
            # CFP blocks cho các feature levels khác nhau
            self.cfp_blocks = nn.ModuleList([
                CFPBlock(embed_dims[0], 256),  # Level 1
                CFPBlock(embed_dims[1], 256),  # Level 2  
                CFPBlock(embed_dims[2], 256),  # Level 3
            ])
            
            # Fusion layer
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(256 * 3 + num_classes, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, 1)
            )
        
        if self.enable_rara:
            # RA-RA blocks
            self.rara_blocks = nn.ModuleList([
                RARABlock(256 if self.enable_cfp else num_classes),
                RARABlock(256 if self.enable_cfp else num_classes),
            ])
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)  # [F1, F2, F3, F4]
        
        # Decoder
        coarse_pred = self.decoder(features)
        
        if not (self.enable_cfp or self.enable_rara):
            # Baseline: Chỉ encoder + decoder
            return coarse_pred
        
        # Refinement
        if self.enable_cfp:
            # CFP processing trên features F1, F2, F3
            cfp_outs = []
            for i, cfp_block in enumerate(self.cfp_blocks):
                cfp_out = cfp_block(features[i])
                # Upsample về kích thước output
                cfp_out = nn.functional.interpolate(
                    cfp_out, size=coarse_pred.shape[-2:], 
                    mode='bilinear', align_corners=False
                )
                cfp_outs.append(cfp_out)
            
            # Fusion
            fusion_input = torch.cat([*cfp_outs, coarse_pred], dim=1)
            refined_pred = self.fusion_conv(fusion_input)
        else:
            refined_pred = coarse_pred
        
        if self.enable_rara:
            # RA-RA processing
            for rara_block in self.rara_blocks:
                refined_pred = rara_block(refined_pred)
        
        # Return multiple outputs for deep supervision
        if self.enable_cfp or self.enable_rara:
            return [refined_pred, coarse_pred]
        else:
            return coarse_pred


def run_ablation_experiment(config_name, model_config, args):
    """
    Chạy một thí nghiệm ablation
    """
    print(f"\n{'='*60}")
    print(f"Running Ablation: {config_name}")
    print(f"{'='*60}")
    print(f"Config: {model_config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = AblationColonFormer(
        backbone=args.backbone,
        num_classes=args.num_classes,
        **model_config
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create data module
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
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create loss, optimizer, scheduler
    criterion = ColonFormerLoss(
        alpha=args.alpha,
        gamma=args.gamma,
        lambda_weight=args.lambda_weight
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"ablation_{config_name}_{timestamp}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        args=args
    )
    
    # Training
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train()
    
    # Final validation
    print(f"\nFinal validation for {config_name}:")
    final_metrics = evaluate_model(model, val_loader, device, criterion)
    print_metrics(final_metrics, f"{config_name} Final")
    
    return final_metrics, save_dir


def main():
    parser = argparse.ArgumentParser(description='ColonFormer Ablation Study')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='mit_b3',
                        choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'],
                        help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes')
    
    # Training parameters (giảm epochs cho ablation study)
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for each ablation')
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
    
    # Ablation parameters
    parser.add_argument('--save_dir', type=str, default='ablation_results',
                        help='Save directory for ablation results')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save interval')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline experiment')
    parser.add_argument('--skip_cfp', action='store_true',
                        help='Skip CFP experiment')
    parser.add_argument('--skip_full', action='store_true',
                        help='Skip full model experiment')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create main save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Define ablation experiments theo Bảng 5
    experiments = []
    
    if not args.skip_baseline:
        experiments.append((
            "Baseline_MiT_UPER",
            {"enable_cfp": False, "enable_rara": False}
        ))
    
    if not args.skip_cfp:
        experiments.append((
            "MiT_UPER_CFP", 
            {"enable_cfp": True, "enable_rara": False}
        ))
    
    if not args.skip_full:
        experiments.append((
            "ColonFormer_Full",
            {"enable_cfp": True, "enable_rara": True}
        ))
    
    # Run experiments
    results = {}
    
    print("ColonFormer Ablation Study")
    print("="*60)
    print(f"Backbone: {args.backbone}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Experiments: {len(experiments)}")
    
    for config_name, model_config in experiments:
        try:
            final_metrics, save_dir = run_ablation_experiment(
                config_name, model_config, args
            )
            results[config_name] = final_metrics
            
        except Exception as e:
            print(f"Error in {config_name}: {e}")
            results[config_name] = None
    
    # Print final comparison
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Configuration':<20} {'mDice':<8} {'mIoU':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 70)
    
    for config_name, metrics in results.items():
        if metrics is not None:
            dice = metrics.get('mDice', 0)
            iou = metrics.get('mIoU', 0) 
            precision = metrics.get('Precision', 0)
            recall = metrics.get('Recall', 0)
            
            print(f"{config_name:<20} {dice:<8.4f} {iou:<8.4f} {precision:<10.4f} {recall:<8.4f}")
        else:
            print(f"{config_name:<20} {'FAILED':<8} {'FAILED':<8} {'FAILED':<10} {'FAILED':<8}")
    
    # Save results
    results_file = os.path.join(args.save_dir, f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(results_file, 'w') as f:
        f.write("ColonFormer Ablation Study Results\n")
        f.write("="*50 + "\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n\n")
        
        f.write(f"{'Configuration':<20} {'mDice':<8} {'mIoU':<8} {'Precision':<10} {'Recall':<8}\n")
        f.write("-" * 70 + "\n")
        
        for config_name, metrics in results.items():
            if metrics is not None:
                dice = metrics.get('mDice', 0)
                iou = metrics.get('mIoU', 0)
                precision = metrics.get('Precision', 0)
                recall = metrics.get('Recall', 0)
                
                f.write(f"{config_name:<20} {dice:<8.4f} {iou:<8.4f} {precision:<10.4f} {recall:<8.4f}\n")
            else:
                f.write(f"{config_name:<20} {'FAILED':<8} {'FAILED':<8} {'FAILED':<10} {'FAILED':<8}\n")
    
    print(f"\nAblation study completed!")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main() 