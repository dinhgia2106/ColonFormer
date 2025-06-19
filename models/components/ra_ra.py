"""
Residual Axial Attention (RA-RA) Block implementation
Một trong những đóng góp chính của ColonFormer
Dựa trên Axial Attention mechanism với residual connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AxialAttention(nn.Module):
    """
    Axial Attention mechanism
    Tính attention theo từng trục (height hoặc width) một cách tuần tự
    """
    def __init__(self, dim, num_heads=8, axis='height', qkv_bias=False, dropout=0.):
        super(AxialAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.axis = axis  # 'height' or 'width'
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: input tensor [B, C, H, W]
        Returns:
            output: attention output [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        if self.axis == 'height':
            # Reshape to [B*W, H, C] for height-wise attention
            x = x.permute(0, 3, 2, 1).contiguous()  # [B, W, H, C]
            x = x.view(B * W, H, C)  # [B*W, H, C]
            N = H
        else:  # width
            # Reshape to [B*H, W, C] for width-wise attention
            x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            x = x.view(B * H, W, C)  # [B*H, W, C]
            N = W
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(-1, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*H_or_W, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(-1, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        # Reshape back to original shape
        if self.axis == 'height':
            x = x.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()  # [B, C, H, W]
        else:  # width
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        return x


class ResidualAxialAttention(nn.Module):
    """
    Residual Axial Attention Block (RA-RA)
    Áp dụng axial attention theo cả height và width với residual connection
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0., norm_layer=nn.LayerNorm):
        super(ResidualAxialAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Height-wise axial attention
        self.height_attn = AxialAttention(
            dim, num_heads=num_heads, axis='height', 
            qkv_bias=qkv_bias, dropout=dropout
        )
        
        # Width-wise axial attention  
        self.width_attn = AxialAttention(
            dim, num_heads=num_heads, axis='width',
            qkv_bias=qkv_bias, dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # Convolution layers for feature refinement
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: input feature map [B, C, H, W]
        Returns:
            output: processed feature map [B, C, H, W]
        """
        identity = x
        
        # Height-wise attention với residual connection
        B, C, H, W = x.shape
        
        # Normalize for height attention
        x_norm = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        # Apply height attention
        h_attn = self.height_attn(x_norm)
        x = x + h_attn  # Residual connection
        
        # Normalize for width attention
        x_norm = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        # Apply width attention
        w_attn = self.width_attn(x_norm)
        x = x + w_attn  # Residual connection
        
        # Additional convolution refinement
        conv_out = self.conv1(x)
        conv_out = self.bn1(conv_out)
        conv_out = self.relu(conv_out)
        
        conv_out = self.conv2(conv_out)
        conv_out = self.bn2(conv_out)
        
        # Final residual connection
        output = identity + conv_out
        output = self.relu(output)
        
        return output


class ImprovedResidualAxialAttention(nn.Module):
    """
    Improved RA-RA block với additional enhancements
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0., 
                 reduction_ratio=4, norm_layer=nn.LayerNorm):
        super(ImprovedResidualAxialAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Axial attention layers
        self.height_attn = AxialAttention(
            dim, num_heads=num_heads, axis='height', 
            qkv_bias=qkv_bias, dropout=dropout
        )
        self.width_attn = AxialAttention(
            dim, num_heads=num_heads, axis='width',
            qkv_bias=qkv_bias, dropout=dropout
        )
        
        # Normalization layers
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction_ratio, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        
        # Skip connection adjustment
        self.skip_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: input feature map [B, C, H, W]
        Returns:
            output: processed feature map [B, C, H, W]
        """
        B, C, H, W = x.shape
        identity = x
        
        # Height-wise attention
        x_h_norm = x.permute(0, 2, 3, 1).contiguous()
        x_h_norm = self.norm1(x_h_norm).permute(0, 3, 1, 2).contiguous()
        h_attn = self.height_attn(x_h_norm)
        x = x + h_attn
        
        # Width-wise attention
        x_w_norm = x.permute(0, 2, 3, 1).contiguous()
        x_w_norm = self.norm2(x_w_norm).permute(0, 3, 1, 2).contiguous()
        w_attn = self.width_attn(x_w_norm)
        x = x + w_attn
        
        # Channel attention
        channel_weights = self.channel_attn(x)
        x = x * channel_weights
        
        # Feature refinement
        refined = self.refine_conv(x)
        
        # Skip connection
        skip = self.skip_conv(identity)
        
        # Final output
        output = skip + refined
        
        return output

    def __repr__(self):
        return f"ImprovedResidualAxialAttention(dim={self.dim}, num_heads={self.num_heads})"


if __name__ == "__main__":
    # Test RA-RA blocks
    x = torch.randn(2, 256, 32, 32)
    
    # Test basic RA-RA
    rara_basic = ResidualAxialAttention(256, num_heads=8)
    out_basic = rara_basic(x)
    print(f"Input shape: {x.shape}")
    print(f"Basic RA-RA output shape: {out_basic.shape}")
    
    # Test improved RA-RA
    rara_improved = ImprovedResidualAxialAttention(256, num_heads=8)
    out_improved = rara_improved(x)
    print(f"Improved RA-RA output shape: {out_improved.shape}")
    
    print(f"Basic RA-RA: {rara_basic}")
    print(f"Improved RA-RA: {rara_improved}") 