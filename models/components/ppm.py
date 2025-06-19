"""
Pyramid Pooling Module (PPM) implementation
Dựa trên PSPNet và được sử dụng trong ColonFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    """
    Pyramid Pooling Module (PPM)
    Thực hiện pooling với nhiều kích thước khác nhau và kết hợp lại
    """
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PyramidPoolingModule(nn.Module):
    """
    Phiên bản cải tiến của PPM cho ColonFormer
    """
    def __init__(self, in_channels, out_channels=None, pool_scales=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        if out_channels is None:
            out_channels = in_channels // len(pool_scales)
        
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Tạo các branch pooling
        self.branches = nn.ModuleList()
        for pool_scale in pool_scales:
            self.branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Conv cuối để kết hợp
        total_channels = in_channels + out_channels * len(pool_scales)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: input feature map [N, C, H, W]
        Returns:
            output: processed feature map [N, C, H, W]
        """
        input_size = x.shape[2:]
        
        # Thu thập outputs từ các branch
        branch_outputs = [x]
        
        for branch in self.branches:
            branch_out = branch(x)
            # Upsample về kích thước gốc
            branch_out = F.interpolate(
                branch_out, 
                size=input_size, 
                mode='bilinear', 
                align_corners=True
            )
            branch_outputs.append(branch_out)
        
        # Concatenate tất cả outputs
        concat_out = torch.cat(branch_outputs, dim=1)
        
        # Fusion conv
        output = self.fusion_conv(concat_out)
        
        return output

    def __repr__(self):
        return f"PyramidPoolingModule(in_channels={self.in_channels}, " \
               f"out_channels={self.out_channels}, pool_scales={self.pool_scales})"


if __name__ == "__main__":
    # Test PPM
    x = torch.randn(2, 512, 32, 32)
    ppm = PyramidPoolingModule(512, 128, pool_scales=(1, 2, 3, 6))
    out = ppm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"PPM: {ppm}") 