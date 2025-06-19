"""
Channel-wise Feature Pyramid (CFP) Block implementation
Một trong những đóng góp chính của ColonFormer
Dựa trên Figure 2(b) trong bài báo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFPBlock(nn.Module):
    """
    Channel-wise Feature Pyramid Block
    Sử dụng nhiều dilation rates để capture multi-scale information
    """
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 3, 5, 9]):
        super(CFPBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_rates = dilation_rates
        
        # Tạo các nhánh với dilation rates khác nhau
        self.branches = nn.ModuleList()
        for dilation_rate in dilation_rates:
            branch = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels // len(dilation_rates), 
                    kernel_size=3, 
                    padding=dilation_rate, 
                    dilation=dilation_rate,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels // len(dilation_rates)),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # Conv 1x1 để điều chỉnh số channels sau concatenation
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Skip connection nếu in_channels != out_channels
        if in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: input feature map [N, C, H, W]
        Returns:
            output: processed feature map [N, out_channels, H, W]
        """
        # Thu thập outputs từ các branch
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # Concatenate outputs từ các branch
        concat_out = torch.cat(branch_outputs, dim=1)
        
        # Fusion conv
        fused_out = self.fusion_conv(concat_out)
        
        # Skip connection
        skip_out = self.skip_conv(x)
        
        # Element-wise addition
        output = fused_out + skip_out
        
        return output

    def __repr__(self):
        return f"CFPBlock(in_channels={self.in_channels}, " \
               f"out_channels={self.out_channels}, dilation_rates={self.dilation_rates})"


class ChannelwiseFeaturePyramid(nn.Module):
    """
    Improved CFP module with better feature fusion
    """
    def __init__(self, in_channels, out_channels=None, dilation_rates=[1, 3, 5, 9], reduction_ratio=4):
        super(ChannelwiseFeaturePyramid, self).__init__()
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_rates = dilation_rates
        
        # Branch channels
        branch_channels = out_channels // len(dilation_rates)
        
        # Dilated convolution branches
        self.branches = nn.ModuleList()
        for dilation_rate in dilation_rates:
            branch = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    branch_channels, 
                    kernel_size=3, 
                    padding=dilation_rate, 
                    dilation=dilation_rate,
                    bias=False
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # Global average pooling branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        total_channels = branch_channels * (len(dilation_rates) + 1)  # +1 for global branch
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: input feature map [N, C, H, W]
        Returns:
            output: processed feature map [N, out_channels, H, W]
        """
        input_size = x.shape[2:]
        
        # Dilated convolution branches
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # Global branch
        global_out = self.global_branch(x)
        global_out = F.interpolate(
            global_out, 
            size=input_size, 
            mode='bilinear', 
            align_corners=True
        )
        branch_outputs.append(global_out)
        
        # Concatenate all branches
        concat_out = torch.cat(branch_outputs, dim=1)
        
        # Feature fusion
        fused_out = self.fusion(concat_out)
        
        # Apply attention
        attention_weights = self.attention(fused_out)
        attended_out = fused_out * attention_weights
        
        # Skip connection
        skip_out = self.skip_conv(x)
        
        # Final output
        output = attended_out + skip_out
        
        return output

    def __repr__(self):
        return f"ChannelwiseFeaturePyramid(in_channels={self.in_channels}, " \
               f"out_channels={self.out_channels}, dilation_rates={self.dilation_rates})"


if __name__ == "__main__":
    # Test CFP blocks
    x = torch.randn(2, 256, 64, 64)
    
    # Test basic CFP
    cfp_basic = CFPBlock(256, 256, dilation_rates=[1, 3, 5, 9])
    out_basic = cfp_basic(x)
    print(f"Input shape: {x.shape}")
    print(f"Basic CFP output shape: {out_basic.shape}")
    
    # Test improved CFP
    cfp_improved = ChannelwiseFeaturePyramid(256, 256, dilation_rates=[1, 3, 5, 9])
    out_improved = cfp_improved(x)
    print(f"Improved CFP output shape: {out_improved.shape}")
    
    print(f"Basic CFP: {cfp_basic}")
    print(f"Improved CFP: {cfp_improved}") 