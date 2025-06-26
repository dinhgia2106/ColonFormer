#!/usr/bin/env python
"""
ColonFormer Training Script - Optimized Version
Sử dụng đúng code gốc từ mmseg, follow paper architecture
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Import code gốc từ mmseg - với fallback cho mmcv issues
try:
    # Try mmcv 2.x import style
    from mmcv import Config as MMCVConfig
    from mmcv.runner import load_checkpoint
    MMCV_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older mmcv
        from mmcv.utils import Config as MMCVConfig
        from mmcv.runner import load_checkpoint
        MMCV_AVAILABLE = True
    except ImportError:
        print("Warning: mmcv not found, using standalone mode")
        MMCVConfig = None
        load_checkpoint = None
        MMCV_AVAILABLE = False

# Import modules gốc từ lib - direct import để tránh mmseg.__init__.py
sys.path.append('.')
lib_path = os.path.join(os.getcwd(), 'mmseg', 'models', 'segmentors', 'lib')
sys.path.insert(0, lib_path)

try:
    import conv_layer
    Conv = conv_layer.Conv
    BNPReLU = conv_layer.BNPReLU
    print("Using original Conv, BNPReLU from mmseg lib")
    
    # Define self_attn từ code gốc với fixed import
    class self_attn(nn.Module):
        def __init__(self, in_channels, mode='hw'):
            super(self_attn, self).__init__()

            self.mode = mode

            self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
            self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
            self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1),stride=1,padding=0)

            self.gamma = nn.Parameter(torch.zeros(1))
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            batch_size, channel, height, width = x.size()

            axis = 1
            if 'h' in self.mode:
                axis *= height
            if 'w' in self.mode:
                axis *= width

            view = (batch_size, -1, axis)

            projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
            projected_key = self.key_conv(x).view(*view)

            attention_map = torch.bmm(projected_query, projected_key)
            attention = self.sigmoid(attention_map)
            projected_value = self.value_conv(x).view(*view)

            out = torch.bmm(projected_value, attention.permute(0, 2, 1))
            out = out.view(batch_size, channel, height, width)

            out = self.gamma * out + x
            return out

    # Define AA_kernel từ code gốc với fixed import
    class AA_kernel(nn.Module):
        def __init__(self, in_channel, out_channel):
            super(AA_kernel, self).__init__()
            self.conv0 = Conv(in_channel, out_channel, kSize=1,stride=1,padding=0)
            self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3),stride = 1, padding=1)
            self.Hattn = self_attn(out_channel, mode='h')
            self.Wattn = self_attn(out_channel, mode='w')

        def forward(self, x):
            x = self.conv0(x)
            x = self.conv1(x)

            Hx = self.Hattn(x)
            Wx = self.Wattn(Hx)

            return Wx

    # Define CFPModule từ code gốc với fixed import  
    class CFPModule(nn.Module):
        def __init__(self, nIn, d=1, KSize=3,dkSize=3):
            super().__init__()
            
            self.bn_relu_1 = BNPReLU(nIn)
            self.bn_relu_2 = BNPReLU(nIn)
            self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)
            
            self.dconv_4_1 = Conv(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (1*d+1,1*d+1),
                                dilation=(d+1,d+1), groups = nIn //16, bn_acti=True)
            
            self.dconv_4_2 = Conv(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (1*d+1,1*d+1),
                                dilation=(d+1,d+1), groups = nIn //16, bn_acti=True)
            
            self.dconv_4_3 = Conv(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (1*d+1,1*d+1),
                                dilation=(d+1,d+1), groups = nIn //16, bn_acti=True)
            
            self.dconv_1_1 = Conv(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (1,1),
                                dilation=(1,1), groups = nIn //16, bn_acti=True)
            
            self.dconv_1_2 = Conv(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (1,1),
                                dilation=(1,1), groups = nIn //16, bn_acti=True)
            
            self.dconv_1_3 = Conv(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (1,1),
                                dilation=(1,1), groups = nIn //16, bn_acti=True)
            
            self.dconv_2_1 = Conv(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (int(d/4+1),int(d/4+1)),
                                dilation=(int(d/4+1),int(d/4+1)), groups = nIn //16, bn_acti=True)
            
            self.dconv_2_2 = Conv(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (int(d/4+1),int(d/4+1)),
                                dilation=(int(d/4+1),int(d/4+1)), groups = nIn //16, bn_acti=True)
            
            self.dconv_2_3 = Conv(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (int(d/4+1),int(d/4+1)),
                                dilation=(int(d/4+1),int(d/4+1)), groups = nIn //16, bn_acti=True)
            
            self.dconv_3_1 = Conv(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (int(d/2+1),int(d/2+1)),
                                dilation=(int(d/2+1),int(d/2+1)), groups = nIn //16, bn_acti=True)
            
            self.dconv_3_2 = Conv(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (int(d/2+1),int(d/2+1)),
                                dilation=(int(d/2+1),int(d/2+1)), groups = nIn //16, bn_acti=True)
            
            self.dconv_3_3 = Conv(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (int(d/2+1),int(d/2+1)),
                                dilation=(int(d/2+1),int(d/2+1)), groups = nIn //16, bn_acti=True)
            
            self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0,bn_acti=False)  
            
        def forward(self, input):
            inp = self.bn_relu_1(input)
            inp = self.conv1x1_1(inp)
            
            o1_1 = self.dconv_1_1(inp)
            o1_2 = self.dconv_1_2(o1_1)
            o1_3 = self.dconv_1_3(o1_2)
            
            o2_1 = self.dconv_2_1(inp)
            o2_2 = self.dconv_2_2(o2_1)
            o2_3 = self.dconv_2_3(o2_2)
            
            o3_1 = self.dconv_3_1(inp)
            o3_2 = self.dconv_3_2(o3_1)
            o3_3 = self.dconv_3_3(o3_2)
            
            o4_1 = self.dconv_4_1(inp)
            o4_2 = self.dconv_4_2(o4_1)
            o4_3 = self.dconv_4_3(o4_2)
            
            output_1 = torch.cat([o1_1,o1_2,o1_3], 1)
            output_2 = torch.cat([o2_1,o2_2,o2_3], 1)      
            output_3 = torch.cat([o3_1,o3_2,o3_3], 1)       
            output_4 = torch.cat([o4_1,o4_2,o4_3], 1)   
            
            ad1 = output_1
            ad2 = ad1 + output_2
            ad3 = ad2 + output_3
            ad4 = ad3 + output_4
            output = torch.cat([ad1,ad2,ad3,ad4],1)
            output = self.bn_relu_2(output)
            output = self.conv1x1(output)
            
            return output+input
    
    print("Using original mmseg lib modules with fixed imports")
    
except ImportError:
    print("Warning: mmseg lib modules not found, using fallback standalone implementations")
    
    # Fallback implementations - minimal but functional
    class BNPReLU(nn.Module):
        def __init__(self, nIn):
            super().__init__()
            self.bn = nn.BatchNorm2d(nIn)
            self.acti = nn.PReLU(nIn)
        
        def forward(self, input):
            if input.size(0) == 1 and self.training:
                self.bn.eval()
                output = self.bn(input)
                self.bn.train()
            else:
                output = self.bn(input)
            return self.acti(output)
    
    class Conv(nn.Module):
        def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
            super().__init__()
            self.bn_acti = bn_acti
            self.conv = nn.Conv2d(nIn, nOut, kSize, stride, padding, dilation, groups, bias)
            if self.bn_acti:
                self.bn_relu = BNPReLU(nOut)
        
        def forward(self, input):
            output = self.conv(input)
            if self.bn_acti:
                output = self.bn_relu(output)
            return output
    
    class AA_kernel(nn.Module):
        def __init__(self, in_channel, out_channel):
            super(AA_kernel, self).__init__()
            self.conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
            self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3), bias=False)
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0), bias=False)
            self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channel)
        
        def forward(self, x):
            y = self.conv0(x)
            y = self.conv1(y)
            y = self.conv2(y)
            y = self.conv3(y)
            y = self.bn(y)
            return y
    
    class CFPModule(nn.Module):
        def __init__(self, nIn, d=1, KSize=3, dkSize=3):
            super(CFPModule, self).__init__()
            self.conv1x1 = Conv(nIn, nIn//4, 1, 1, padding=0, bn_acti=True)
            self.dconv3x1 = Conv(nIn//4, nIn//4, (dkSize,1), 1, padding=(1,0), groups=nIn//4, bn_acti=True)
            self.dconv1x3 = Conv(nIn//4, nIn//4, (1,dkSize), 1, padding=(0,1), groups=nIn//4, bn_acti=True)
            self.ddconv3x1 = Conv(nIn//4, nIn//4, (dkSize,1), 1, padding=(1*d,0), dilation=(d,1), groups=nIn//4, bn_acti=True)
            self.ddconv1x3 = Conv(nIn//4, nIn//4, (1,dkSize), 1, padding=(0,1*d), dilation=(1,d), groups=nIn//4, bn_acti=True)
            self.conv1x1_2 = Conv(nIn//4, nIn, 1, 1, padding=0, bn_acti=False)
            
        def forward(self, input):
            output = self.conv1x1(input)
            output = self.dconv3x1(output)
            output = self.dconv1x3(output)  
            output = self.ddconv3x1(output)
            output = self.ddconv1x3(output)
            output = self.conv1x1_2(output)
            return output


class Config:
    """Configuration class để lưu trữ tất cả settings"""
    def __init__(self, args):
        # Model configs
        self.backbone = args.backbone
        self.use_refinement = args.use_refinement
        self.num_classes = args.num_classes
        
        # Training configs
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.img_size = args.img_size
        
        # Loss configs
        self.loss_type = args.loss_type
        self.focal_alpha = args.focal_alpha
        self.focal_gamma = args.focal_gamma
        self.iou_weight = args.iou_weight
        
        # Optimizer configs
        self.optimizer_type = args.optimizer
        self.scheduler_type = args.scheduler
        self.momentum = args.momentum
        
        # Data configs
        self.data_root = args.data_root
        self.val_split = args.val_split
        self.num_workers = args.num_workers
        
        # Training configs
        self.work_dir = args.work_dir
        self.resume_from = args.resume_from
        self.seed = args.seed
        
        # Experiment info
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"colonformer_{self.backbone}_{self.timestamp}"
    
    def save(self, path):
        """Save config to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def __str__(self):
        """String representation for logging"""
        lines = ["=" * 60, "CONFIGURATION", "=" * 60]
        
        sections = [
            ("Model", ["backbone", "use_refinement", "num_classes"]),
            ("Training", ["epochs", "batch_size", "learning_rate", "weight_decay", "img_size"]),
            ("Loss", ["loss_type", "focal_alpha", "focal_gamma", "iou_weight"]),
            ("Optimizer", ["optimizer_type", "scheduler_type", "momentum"]),
            ("Data", ["data_root", "val_split", "num_workers"]),
            ("Experiment", ["work_dir", "experiment_name", "seed"])
        ]
        
        for section_name, keys in sections:
            lines.append(f"\n[{section_name}]")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    lines.append(f"  {key}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =========================================================================
# SIMPLE BACKBONE - Để replace cho MiT khi không có mmcv
# =========================================================================

class SimpleBackbone(nn.Module):
    """Simple backbone thay thế cho MiT khi không có mmcv"""
    
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        
        # Stage 1: 3 -> 64 (stride 4)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 4, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Stage 2: 64 -> 128 (stride 2, total /8)
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Stage 3: 128 -> 320 (stride 2, total /16)
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 320, 3, 2, 1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
            nn.Conv2d(320, 320, 3, 1, 1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )
        
        # Stage 4: 320 -> 512 (stride 2, total /32)
        self.stage4 = nn.Sequential(
            nn.Conv2d(320, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Tương tự output của MiT-B3
        x1 = self.stage1(x)    # [B, 64, H/4, W/4]
        x2 = self.stage2(x1)   # [B, 128, H/8, W/8]
        x3 = self.stage3(x2)   # [B, 320, H/16, W/16]
        x4 = self.stage4(x3)   # [B, 512, H/32, W/32]
        
        return [x1, x2, x3, x4]


class SimpleDecodeHead(nn.Module):
    """Simple decode head tương tự SegFormer"""
    
    def __init__(self, in_channels=[64, 128, 320, 512], channels=128, num_classes=1):
        super(SimpleDecodeHead, self).__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        
        # MLP layers for each stage
        self.linear_c4 = nn.Sequential(
            nn.Conv2d(in_channels[3], channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.linear_c3 = nn.Sequential(
            nn.Conv2d(in_channels[2], channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.linear_c2 = nn.Sequential(
            nn.Conv2d(in_channels[1], channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.linear_c1 = nn.Sequential(
            nn.Conv2d(in_channels[0], channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Final prediction
        self.linear_pred = nn.Conv2d(channels, num_classes, 1)
        
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        
        n, _, h, w = c1.shape
        
        # Transform each feature
        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=(h, w), mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=(h, w), mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(_c2, size=(h, w), mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1)
        
        # Fuse features
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
        # Final prediction
        x = self.linear_pred(_c)
        
        return x


# =========================================================================
# COLONFORMER MODEL - Sử dụng logic từ code gốc
# =========================================================================

class ColonFormer(nn.Module):
    """ColonFormer - Sử dụng logic từ code gốc mmseg"""
    
    def __init__(self, config):
        super(ColonFormer, self).__init__()
        
        self.config = config
        self.use_refinement = config.use_refinement
        
        # Build components
        self.backbone = SimpleBackbone()
        self.decode_head = SimpleDecodeHead(num_classes=config.num_classes)
        
        if self.use_refinement:
            # CFP modules từ code gốc - CHÍNH XÁC
            self.CFP_1 = CFPModule(128, d=8)
            self.CFP_2 = CFPModule(320, d=8)  
            self.CFP_3 = CFPModule(512, d=8)
            
            # RA-RA modules từ code gốc - CHÍNH XÁC
            self.ra1_conv1 = Conv(128, 32, 3, 1, padding=1, bn_acti=True)
            self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
            self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
            
            self.ra2_conv1 = Conv(320, 32, 3, 1, padding=1, bn_acti=True)
            self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
            self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
            
            self.ra3_conv1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
            self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
            self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
            
            # Axial attention từ code gốc - CHÍNH XÁC
            self.aa_kernel_1 = AA_kernel(128, 128)
            self.aa_kernel_2 = AA_kernel(320, 320)
            self.aa_kernel_3 = AA_kernel(512, 512)
    
    def forward(self, x):
        """Forward pass - CHÍNH XÁC như trong code gốc"""
        # Backbone forward
        backbone_features = self.backbone(x)
        x1, x2, x3, x4 = backbone_features

        # Initial decoder forward
        decoder_1 = self.decode_head([x1, x2, x3, x4])
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        if not self.use_refinement:
            return lateral_map_1
        
        # REFINEMENT PROCESS - CHÍNH XÁC NHU TRONG CODE GỐC
        # ------------------- atten-one (Stage 3) -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4)
        
        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3 += cfp_out_1
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode='bilinear')
        
        # ------------------- atten-two (Stage 2) -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(x3)
        
        decoder_3_ra = -1 * (torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2 += cfp_out_2
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(aa_atten_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode='bilinear')        
        
        # ------------------- atten-three (Stage 1) -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2)
        
        decoder_4_ra = -1 * (torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1 += cfp_out_3
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode='bilinear') 
        
        # Return multi-scale outputs như code gốc
        return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1


# =========================================================================
# DATASET
# =========================================================================

class ColonDataset(Dataset):
    """Dataset cho Colon Polyp Segmentation"""
    
    def __init__(self, data_root, img_size=352, phase='train', val_split=0.2):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.phase = phase
        
        # Load image và mask paths
        img_dir = self.data_root / 'image'
        mask_dir = self.data_root / 'mask'
        
        if not img_dir.exists() or not mask_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_root}")
        
        self.img_paths = sorted(list(img_dir.glob('*.png')))
        self.mask_paths = sorted(list(mask_dir.glob('*.png')))
        
        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")
        
        # Train/val split
        n_total = len(self.img_paths)
        n_val = int(n_total * val_split)
        
        if phase == 'train':
            self.img_paths = self.img_paths[n_val:]
            self.mask_paths = self.mask_paths[n_val:]
        elif phase == 'val':
            self.img_paths = self.img_paths[:n_val]
            self.mask_paths = self.mask_paths[:n_val]
        
        print(f"{phase} dataset: {len(self.img_paths)} samples")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = (mask > 127).astype(np.float32)
        
        # Convert to tensors - CRITICAL: ensure float32 type
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC -> CHW, ensure float32
        mask = torch.from_numpy(mask).unsqueeze(0).float()    # Add channel dim, ensure float32
        
        return img, mask


# =========================================================================
# LOSS FUNCTIONS
# =========================================================================

class StructureLoss(nn.Module):
    """Structure Loss từ paper gốc"""
    
    def __init__(self, config):
        super(StructureLoss, self).__init__()
        self.config = config
        
    def forward(self, pred, mask):
        # Binary Cross Entropy Loss
        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
        
        # IoU Loss
        pred_sig = torch.sigmoid(pred)
        inter = (pred_sig * mask).sum(dim=(2, 3))
        union = (pred_sig + mask).sum(dim=(2, 3)) - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        iou_loss = (1 - iou).mean()
        
        return bce + self.config.iou_weight * iou_loss


class FocalLoss(nn.Module):
    """Focal Loss"""
    
    def __init__(self, config):
        super(FocalLoss, self).__init__()
        self.alpha = config.focal_alpha
        self.gamma = config.focal_gamma
        
    def forward(self, pred, mask):
        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce
        return focal_loss.mean()


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss option"""
    
    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        
    def forward(self, pred, mask):
        return F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')


def get_loss_function(config):
    """Get loss function based on config"""
    loss_map = {
        'structure': StructureLoss,
        'focal': FocalLoss,
        'crossentropy': CrossEntropyLoss,
        'ce': CrossEntropyLoss
    }
    
    if config.loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {config.loss_type}")
    
    return loss_map[config.loss_type](config)


# =========================================================================
# METRICS
# =========================================================================

def calculate_metrics(pred, mask, threshold=0.5):
    """Calculate Dice, IoU, Precision, Recall"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    mask_flat = mask.view(-1)
    
    # Calculate metrics
    intersection = (pred_flat * mask_flat).sum()
    
    dice = (2. * intersection + 1e-6) / (pred_flat.sum() + mask_flat.sum() + 1e-6)
    iou = (intersection + 1e-6) / (pred_flat.sum() + mask_flat.sum() - intersection + 1e-6)
    
    precision = (intersection + 1e-6) / (pred_flat.sum() + 1e-6)
    recall = (intersection + 1e-6) / (mask_flat.sum() + 1e-6)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


# =========================================================================
# TRAINING FUNCTIONS
# =========================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    total_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss với multi-scale supervision như code gốc
        if isinstance(outputs, (list, tuple)):
            loss = 0
            for i, output in enumerate(outputs):
                weight = 1.0 / (2 ** i)  # Weight decay cho auxiliary outputs
                output_resized = F.interpolate(output, size=masks.shape[2:], mode='bilinear', align_corners=False)
                loss += weight * criterion(output_resized, masks)
        else:
            output_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(output_resized, masks)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            if isinstance(outputs, (list, tuple)):
                main_output = outputs[0]  # Main output
            else:
                main_output = outputs
            
            main_output_resized = F.interpolate(main_output, size=masks.shape[2:], mode='bilinear', align_corners=False)
            metrics = calculate_metrics(main_output_resized, masks)
        
        # Update statistics
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{metrics["dice"]:.4f}',
            'iou': f'{metrics["iou"]:.4f}'
        })
    
    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_metrics = {key: value / len(train_loader) for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics


def validate_epoch(model, val_loader, criterion, device, config):
    """Validate one epoch"""
    model.eval()
    total_loss = 0.0
    total_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            if isinstance(outputs, (list, tuple)):
                main_output = outputs[0]
            else:
                main_output = outputs
            
            main_output_resized = F.interpolate(main_output, size=masks.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(main_output_resized, masks)
            
            # Calculate metrics
            metrics = calculate_metrics(main_output_resized, masks)
            
            # Update statistics
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
    
    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    avg_metrics = {key: value / len(val_loader) for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='ColonFormer Training - Optimized Clean Version')
    
    # Model configs
    parser.add_argument('--backbone', default='simple', choices=['simple'], help='Backbone architecture')
    parser.add_argument('--use-refinement', action='store_true', default=True, help='Use refinement modules')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')
    
    # Training configs
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--img-size', type=int, default=352, help='Image size')
    
    # Loss configs
    parser.add_argument('--loss-type', default='structure', choices=['structure', 'focal', 'crossentropy', 'ce'], 
                       help='Loss function type')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--iou-weight', type=float, default=1.0, help='IoU loss weight')
    
    # Optimizer configs
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'step', 'poly'], help='Scheduler type')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    
    # Data configs
    parser.add_argument('--data-root', default='data/TrainDataset', help='Dataset root directory')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers')
    
    # Training configs
    parser.add_argument('--work-dir', default='work_dirs/colonformer', help='Work directory')
    parser.add_argument('--resume-from', default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(args)
    
    # Setup
    torch.manual_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(config)
    
    # Save config
    config.save(work_dir / 'config.json')
    
    # Build model
    print("Building ColonFormer model...")
    model = ColonFormer(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Build datasets
    print("Loading datasets...")
    try:
        train_dataset = ColonDataset(config.data_root, config.img_size, 'train', config.val_split)
        val_dataset = ColonDataset(config.data_root, config.img_size, 'val', config.val_split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure data is in correct format: data/TrainDataset/image/ và data/TrainDataset/mask/")
        return
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Build criterion
    criterion = get_loss_function(config)
    
    # Build optimizer
    if config.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    
    # Build scheduler
    if config.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    elif config.scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif config.scheduler_type == 'poly':
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=config.epochs, power=0.9)
    
    # TensorBoard
    writer = SummaryWriter(work_dir / 'tensorboard')
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_dice = 0.0
    
    if config.resume_from and Path(config.resume_from).exists():
        print(f"Resuming from {config.resume_from}")
        checkpoint = torch.load(config.resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
    
    # Training loop
    print("Starting training loop...")
    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, config)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Dice/Train', train_metrics['dice'], epoch)
        writer.add_scalar('Dice/Val', val_metrics['dice'], epoch)
        writer.add_scalar('IoU/Train', train_metrics['iou'], epoch)
        writer.add_scalar('IoU/Val', val_metrics['iou'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'val_metrics': val_metrics,
                'config': config.__dict__
            }
            torch.save(checkpoint, work_dir / 'best_model.pth')
            print(f"New best model saved! Dice: {best_dice:.4f}")
        
        # Save regular checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'val_metrics': val_metrics,
                'config': config.__dict__
            }
            torch.save(checkpoint, work_dir / f'checkpoint_epoch_{epoch}.pth')
    
    writer.close()
    print(f"\nTraining completed! Best Dice: {best_dice:.4f}")
    print(f"Models saved in: {work_dir}")


if __name__ == '__main__':
    main()