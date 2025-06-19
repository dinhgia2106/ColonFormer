"""
UPER Decoder implementation từ UPerNet
Được sử dụng trong ColonFormer để decode multi-scale features từ MiT encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ppm import PyramidPoolingModule


class ConvModule(nn.Module):
    """Basic convolution module với norm và activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, bias=False
        )
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True) if act_layer else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class UPerDecoder(nn.Module):
    """
    UPerNet Decoder
    Feature Pyramid Network (FPN) với Pyramid Pooling Module (PPM)
    """
    def __init__(self, in_channels_list, out_channels=256, num_classes=1, 
                 pool_scales=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(UPerDecoder, self).__init__()
        self.in_channels_list = in_channels_list  # [64, 128, 320, 512] cho MiT-B3
        self.out_channels = out_channels
        self.num_classes = num_classes
        
        # PPM cho feature map có resolution thấp nhất (deepest)
        self.ppm = PyramidPoolingModule(
            in_channels=in_channels_list[-1],  # 512
            out_channels=out_channels // len(pool_scales),  # 64
            pool_scales=pool_scales
        )
        
        # Lateral convolutions để điều chỉnh channels
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = ConvModule(
                in_channels, out_channels, kernel_size=1, 
                padding=0, norm_layer=norm_layer
            )
            self.lateral_convs.append(lateral_conv)
        
        # FPN convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            fpn_conv = ConvModule(
                out_channels, out_channels, kernel_size=3,
                padding=1, norm_layer=norm_layer
            )
            self.fpn_convs.append(fpn_conv)
        
        # Final fusion conv
        self.fusion_conv = ConvModule(
            out_channels * len(in_channels_list), out_channels,
            kernel_size=3, padding=1, norm_layer=norm_layer
        )
        
        # Classification head
        self.cls_head = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features):
        """
        Args:
            features: list of feature maps from encoder [F1, F2, F3, F4]
                     với resolutions giảm dần (F1 có resolution cao nhất)
        Returns:
            output: segmentation prediction
            fpn_features: FPN features để sử dụng trong refinement module
        """
        # Apply PPM to deepest feature
        features = list(features)
        features[-1] = self.ppm(features[-1])
        
        # Build FPN laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(features[i]))
        
        # Build FPN top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=True
            )
        
        # Build outputs
        fpn_outs = []
        for i in range(used_backbone_levels):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))
        
        # Resize all feature maps to the same size (highest resolution)
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_size, mode='bilinear', align_corners=True
            )
        
        # Fusion
        concat_features = torch.cat(fpn_outs, dim=1)
        fused_features = self.fusion_conv(concat_features)
        fused_features = self.dropout(fused_features)
        
        # Classification
        output = self.cls_head(fused_features)
        
        return output, fpn_outs

    def __repr__(self):
        return f"UPerDecoder(in_channels={self.in_channels_list}, " \
               f"out_channels={self.out_channels}, num_classes={self.num_classes})"


class ImprovedUPerDecoder(nn.Module):
    """
    Improved UPer Decoder với additional features
    """
    def __init__(self, in_channels_list, out_channels=256, num_classes=1,
                 pool_scales=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d,
                 use_se_block=True, deep_supervision=True):
        super(ImprovedUPerDecoder, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        # PPM
        self.ppm = PyramidPoolingModule(
            in_channels=in_channels_list[-1],
            out_channels=out_channels // len(pool_scales),
            pool_scales=pool_scales
        )
        
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = ConvModule(
                in_channels, out_channels, kernel_size=1,
                padding=0, norm_layer=norm_layer
            )
            self.lateral_convs.append(lateral_conv)
        
        # SE blocks cho attention
        if use_se_block:
            self.se_blocks = nn.ModuleList()
            for _ in range(len(in_channels_list)):
                se_block = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
                    nn.Sigmoid()
                )
                self.se_blocks.append(se_block)
        else:
            self.se_blocks = None
        
        # FPN convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            fpn_conv = ConvModule(
                out_channels, out_channels, kernel_size=3,
                padding=1, norm_layer=norm_layer
            )
            self.fpn_convs.append(fpn_conv)
        
        # Multi-scale fusion
        self.fusion_conv = ConvModule(
            out_channels * len(in_channels_list), out_channels,
            kernel_size=3, padding=1, norm_layer=norm_layer
        )
        
        # Classification heads
        self.cls_head = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        
        # Deep supervision heads nếu được enable
        if deep_supervision:
            self.aux_heads = nn.ModuleList()
            for _ in range(len(in_channels_list) - 1):  # Không include feature cuối
                aux_head = nn.Sequential(
                    ConvModule(out_channels, out_channels // 2, kernel_size=3,
                              padding=1, norm_layer=norm_layer),
                    nn.Conv2d(out_channels // 2, num_classes, kernel_size=1)
                )
                self.aux_heads.append(aux_head)
        
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features):
        """
        Args:
            features: list of feature maps [F1, F2, F3, F4]
        Returns:
            main_output: main segmentation prediction
            aux_outputs: auxiliary outputs for deep supervision (if enabled)
            fpn_features: FPN features
        """
        features = list(features)
        
        # Apply PPM to deepest feature
        features[-1] = self.ppm(features[-1])
        
        # Lateral connections
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            lateral = lateral_conv(features[i])
            
            # Apply SE attention if available
            if self.se_blocks is not None:
                se_weight = self.se_blocks[i](lateral)
                lateral = lateral * se_weight
                
            laterals.append(lateral)
        
        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=True
            )
        
        # FPN outputs
        fpn_outs = []
        aux_outputs = []
        
        for i in range(len(laterals)):
            fpn_out = self.fpn_convs[i](laterals[i])
            fpn_outs.append(fpn_out)
            
            # Auxiliary outputs for deep supervision
            if self.deep_supervision and i < len(laterals) - 1:
                aux_out = self.aux_heads[i](fpn_out)
                aux_outputs.append(aux_out)
        
        # Resize to same resolution
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=target_size, mode='bilinear', align_corners=True
            )
        
        # Fusion
        concat_features = torch.cat(fpn_outs, dim=1)
        fused_features = self.fusion_conv(concat_features)
        fused_features = self.dropout(fused_features)
        
        # Main output
        main_output = self.cls_head(fused_features)
        
        if self.deep_supervision:
            return main_output, aux_outputs, fpn_outs
        else:
            return main_output, fpn_outs

    def __repr__(self):
        return f"ImprovedUPerDecoder(in_channels={self.in_channels_list}, " \
               f"out_channels={self.out_channels}, deep_supervision={self.deep_supervision})"


if __name__ == "__main__":
    # Test UPer Decoder
    # Giả lập features từ MiT-B3: [64, 128, 320, 512] channels
    features = [
        torch.randn(2, 64, 88, 88),   # F1: 1/4 resolution
        torch.randn(2, 128, 44, 44),  # F2: 1/8 resolution  
        torch.randn(2, 320, 22, 22),  # F3: 1/16 resolution
        torch.randn(2, 512, 11, 11)   # F4: 1/32 resolution
    ]
    
    # Test basic UPer Decoder
    decoder = UPerDecoder([64, 128, 320, 512], out_channels=256, num_classes=1)
    main_out, fpn_features = decoder(features)
    print(f"Main output shape: {main_out.shape}")
    print(f"Number of FPN features: {len(fpn_features)}")
    
    # Test improved UPer Decoder
    improved_decoder = ImprovedUPerDecoder(
        [64, 128, 320, 512], out_channels=256, num_classes=1, deep_supervision=True
    )
    main_out, aux_outs, fpn_features = improved_decoder(features)
    print(f"Improved main output shape: {main_out.shape}")
    print(f"Number of auxiliary outputs: {len(aux_outs)}")
    print(f"Auxiliary output shapes: {[aux.shape for aux in aux_outs]}")
    
    print(f"Decoder: {decoder}")
    print(f"Improved Decoder: {improved_decoder}") 