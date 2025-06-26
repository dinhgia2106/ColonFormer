import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import numpy as np
import cv2

from .lib.conv_layer import Conv, BNPReLU
from .lib.axial_atten import AA_kernel
from .lib.context_module import CFPModule

@SEGMENTORS.register_module()
class ColonFormer(BaseSegmentor):
    """ColonFormer segmentor.

    ColonFormer: Efficient Transformer based Architecture for Clinical Polyp Segmentation
    This segmentor consists of backbone, decode_head, and CFP+RA-RA refinement modules.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(ColonFormer, self).__init__()
        
        # Build backbone and decode head
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        # Initialize weights
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        # CFP modules for multi-scale context modeling
        self.CFP_1 = CFPModule(128, d=8)
        self.CFP_2 = CFPModule(320, d=8)
        self.CFP_3 = CFPModule(512, d=8)
        
        # RA-RA (Reverse Attention - Residual Attention) modules
        self.ra1_conv1 = Conv(128, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        self.ra2_conv1 = Conv(320, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        self.ra3_conv1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        # Axial attention kernels
        self.aa_kernel_1 = AA_kernel(128, 128)
        self.aa_kernel_2 = AA_kernel(320, 320)
        self.aa_kernel_3 = AA_kernel(512, 512)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in training."""
        losses = dict()
        
        # Forward through ColonFormer refinement modules
        refined_outputs = self.forward_refinement(x)
        
        # Calculate losses for multi-scale outputs (deep supervision)
        loss_decode = []
        for i, output in enumerate(refined_outputs):
            # Resize output to match ground truth
            output_resized = resize(
                input=output,
                size=gt_semantic_seg.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            
            # Calculate loss with weight decay for auxiliary outputs
            weight = 1.0 / (2 ** i) if i > 0 else 1.0
            loss = self.decode_head.losses(output_resized, gt_semantic_seg)
            
            for loss_name, loss_value in loss.items():
                loss_decode.append(weight * loss_value)
        
        # Sum all losses
        total_loss = sum(loss_decode)
        losses.update(add_prefix({'loss_seg': total_loss}, 'decode'))
        
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and return the main output for testing."""
        refined_outputs = self.forward_refinement(x)
        # Return the main output (highest resolution)
        return refined_outputs[0]

    def forward_refinement(self, x):
        """Forward function with CFP and RA-RA refinement modules."""
        x1 = x[0]  # 64x88x88
        x2 = x[1]  # 128x44x44
        x3 = x[2]  # 320x22x22
        x4 = x[3]  # 512x11x11

        # Initial decode head forward
        decoder_1 = self.decode_head.forward([x1, x2, x3, x4])  # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one (Stage 3) -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4)
        
        # Reverse attention mechanism
        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3 += cfp_out_1
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3)
        
        # Residual attention
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
        
        # Return multi-scale outputs for deep supervision
        return [lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1]

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.
        
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture support semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        losses = dict()
        
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        
        return losses

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.encode_decode(img, img_meta)
        seg_pred = seg_logit.argmax(dim=1)
        if rescale:
            seg_pred = resize(
                seg_pred.float(),
                size=img_meta[0]['ori_shape'][:2],
                mode='nearest')
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.
        
        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.encode_decode(imgs[0], img_metas[0])
        for i in range(1, len(imgs)):
            cur_seg_logit = self.encode_decode(imgs[i], img_metas[i])
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = resize(
            seg_pred.float(),
            size=img_metas[0][0]['ori_shape'][:2],
            mode='nearest')
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred