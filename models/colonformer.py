"""
ColonFormer: Mô hình hoàn chỉnh cho Colon Polyp Segmentation
Kết hợp MiT Encoder, UPer Decoder, và Refinement Module (CFP + RA-RA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import các components đã implement
from .backbones.mit import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from .components.uper_decoder import ImprovedUPerDecoder
from .components.cfp import ChannelwiseFeaturePyramid
from .components.ra_ra import ImprovedResidualAxialAttention


class RefinementModule(nn.Module):
    """
    Refinement Module kết hợp CFP và RA-RA blocks
    """
    def __init__(self, in_channels_list, out_channels=256, num_refine_blocks=3):
        super(RefinementModule, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_refine_blocks = num_refine_blocks
        
        # CFP blocks cho từng level của FPN features
        self.cfp_blocks = nn.ModuleList()
        for in_channels in in_channels_list[:-1]:  # Không áp dụng cho level cuối cùng
            cfp = ChannelwiseFeaturePyramid(
                in_channels=out_channels,  # Sau khi đi qua UPer decoder đã được chuẩn hóa
                out_channels=out_channels,
                dilation_rates=[1, 3, 5, 9]
            )
            self.cfp_blocks.append(cfp)
        
        # RA-RA blocks để refine features
        self.rara_blocks = nn.ModuleList()
        for _ in range(num_refine_blocks):
            rara = ImprovedResidualAxialAttention(
                dim=out_channels,
                num_heads=8,
                dropout=0.1
            )
            self.rara_blocks.append(rara)
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final refinement
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1)  # Output segmentation
        )
        
    def forward(self, coarse_prediction, fpn_features):
        """
        Args:
            coarse_prediction: output từ decoder [B, 1, H, W]
            fpn_features: list of FPN features [F1, F2, F3, F4] mỗi cái [B, out_channels, H_i, W_i]
        
        Returns:
            refined_prediction: [B, 1, H, W]
        """
        target_size = coarse_prediction.shape[2:]
        refined_features = []
        
        # Apply CFP to FPN features (except the last one)
        for i, cfp_block in enumerate(self.cfp_blocks):
            fpn_feat = fpn_features[i]
            
            # Resize to target size
            if fpn_feat.shape[2:] != target_size:
                fpn_feat = F.interpolate(
                    fpn_feat, size=target_size, mode='bilinear', align_corners=True
                )
            
            # Apply CFP
            cfp_out = cfp_block(fpn_feat)
            refined_features.append(cfp_out)
        
        # Add the last FPN feature (without CFP)
        last_fpn = fpn_features[-1]
        if last_fpn.shape[2:] != target_size:
            last_fpn = F.interpolate(
                last_fpn, size=target_size, mode='bilinear', align_corners=True
            )
        refined_features.append(last_fpn)
        
        # Concatenate all refined features
        concat_features = torch.cat(refined_features, dim=1)
        fused_features = self.fusion_conv(concat_features)
        
        # Apply RA-RA blocks
        x = fused_features
        for rara_block in self.rara_blocks:
            x = rara_block(x)
        
        # Final prediction
        refined_prediction = self.final_conv(x)
        
        return refined_prediction


class ColonFormer(nn.Module):
    """
    ColonFormer: Complete model cho polyp segmentation
    """
    def __init__(self, backbone='mit_b3', num_classes=1, img_size=352, 
                 decoder_channels=256, deep_supervision=True, use_refinement=True):
        super(ColonFormer, self).__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.img_size = img_size
        self.deep_supervision = deep_supervision
        self.use_refinement = use_refinement
        
        # Encoder (MiT backbone)
        self.encoder = self._build_encoder(backbone)
        
        # Get encoder output channels
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            encoder_features = self.encoder(dummy_input)
            self.encoder_channels = [feat.shape[1] for feat in encoder_features]
        
        # Decoder (UPer decoder với deep supervision)
        self.decoder = ImprovedUPerDecoder(
            in_channels_list=self.encoder_channels,
            out_channels=decoder_channels,
            num_classes=num_classes,
            deep_supervision=deep_supervision
        )
        
        # Refinement module (CFP + RA-RA)
        if use_refinement:
            self.refinement = RefinementModule(
                in_channels_list=self.encoder_channels,
                out_channels=decoder_channels,
                num_refine_blocks=3
            )
        else:
            self.refinement = None
    
    def _build_encoder(self, backbone):
        """Build MiT encoder"""
        backbone_dict = {
            'mit_b0': mit_b0,
            'mit_b1': mit_b1,
            'mit_b2': mit_b2,
            'mit_b3': mit_b3,
            'mit_b4': mit_b4,
            'mit_b5': mit_b5
        }
        
        if backbone not in backbone_dict:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return backbone_dict[backbone]()
    
    def forward(self, x):
        """
        Args:
            x: input images [B, 3, H, W]
        
        Returns:
            if training và deep_supervision:
                dict với 'main', 'coarse', 'aux' predictions
            else:
                main prediction [B, num_classes, H, W]
        """
        input_size = x.shape[2:]
        
        # Encoder forward
        encoder_features = self.encoder(x)  # [F1, F2, F3, F4]
        
        # Decoder forward
        if self.deep_supervision:
            coarse_pred, aux_outputs, fpn_features = self.decoder(encoder_features)
        else:
            coarse_pred, fpn_features = self.decoder(encoder_features)
            aux_outputs = []
        
        # Resize coarse prediction to input size
        if coarse_pred.shape[2:] != input_size:
            coarse_pred = F.interpolate(
                coarse_pred, size=input_size, mode='bilinear', align_corners=True
            )
        
        # Refinement module
        if self.use_refinement and self.refinement is not None:
            refined_pred = self.refinement(coarse_pred, fpn_features)
            
            # Resize refined prediction to input size
            if refined_pred.shape[2:] != input_size:
                refined_pred = F.interpolate(
                    refined_pred, size=input_size, mode='bilinear', align_corners=True
                )
            
            main_pred = refined_pred
        else:
            main_pred = coarse_pred
        
        # Return format
        if self.training and self.deep_supervision:
            # Resize auxiliary outputs
            resized_aux = []
            for aux_out in aux_outputs:
                if aux_out.shape[2:] != input_size:
                    aux_out = F.interpolate(
                        aux_out, size=input_size, mode='bilinear', align_corners=True
                    )
                resized_aux.append(aux_out)
            
            return {
                'main': main_pred,
                'coarse': coarse_pred,
                'aux': resized_aux
            }
        else:
            return main_pred
    
    def __repr__(self):
        return f"ColonFormer(backbone={self.backbone_name}, " \
               f"num_classes={self.num_classes}, img_size={self.img_size}, " \
               f"deep_supervision={self.deep_supervision}, use_refinement={self.use_refinement})"


def colonformer_s(num_classes=1, img_size=352, **kwargs):
    """ColonFormer-S (Small) với MiT-B1"""
    return ColonFormer(
        backbone='mit_b1', 
        num_classes=num_classes, 
        img_size=img_size, 
        decoder_channels=256,
        **kwargs
    )


def colonformer_l(num_classes=1, img_size=352, **kwargs):
    """ColonFormer-L (Large) với MiT-B3"""
    return ColonFormer(
        backbone='mit_b3', 
        num_classes=num_classes, 
        img_size=img_size, 
        decoder_channels=256,
        **kwargs
    )


def colonformer_xl(num_classes=1, img_size=352, **kwargs):
    """ColonFormer-XL (Extra Large) với MiT-B5"""
    return ColonFormer(
        backbone='mit_b5', 
        num_classes=num_classes, 
        img_size=img_size, 
        decoder_channels=512,
        **kwargs
    )


if __name__ == "__main__":
    # Test ColonFormer models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test input
    batch_size = 2
    img_size = 352
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    print("Testing ColonFormer models...")
    
    # Test ColonFormer-S
    print("\n1. ColonFormer-S (MiT-B1):")
    model_s = colonformer_s(num_classes=1, img_size=img_size).to(device)
    
    # Training mode (with deep supervision)
    model_s.train()
    outputs_train = model_s(x)
    print(f"Training outputs: {type(outputs_train)}")
    if isinstance(outputs_train, dict):
        for key, value in outputs_train.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} predictions, shapes: {[v.shape for v in value]}")
            else:
                print(f"  {key}: {value.shape}")
    
    # Eval mode
    model_s.eval()
    with torch.no_grad():
        outputs_eval = model_s(x)
        print(f"Eval output shape: {outputs_eval.shape}")
    
    # Test ColonFormer-L
    print("\n2. ColonFormer-L (MiT-B3):")
    model_l = colonformer_l(num_classes=1, img_size=img_size).to(device)
    
    model_l.eval()
    with torch.no_grad():
        outputs_l = model_l(x)
        print(f"ColonFormer-L output shape: {outputs_l.shape}")
    
    # Model info
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"ColonFormer-S: {count_parameters(model_s):,} parameters")
    print(f"ColonFormer-L: {count_parameters(model_l):,} parameters")
    
    print(f"\nModel representations:")
    print(f"ColonFormer-S: {model_s}")
    print(f"ColonFormer-L: {model_l}") 