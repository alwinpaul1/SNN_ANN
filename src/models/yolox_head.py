"""
YOLOX Detection Head

Paper uses YOLOX framework for detection with decoupled head design.
This is a clean implementation following YOLOX principles.

Features:
- Decoupled classification and regression branches
- Multi-scale feature processing (P3, P4, P5)
- Anchor-free detection

Paper Reference: Ahmed et al. (arXiv:2403.10173v4)
YOLOX Reference: Ge et al., "YOLOX: Exceeding YOLO Series in 2021"
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class ConvBNReLU(nn.Module):
    """Conv2D + BatchNorm + ReLU block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class YOLOXHead(nn.Module):
    """
    YOLOX Detection Head with decoupled branches.
    
    Architecture per scale:
    - Stem: 1×1 conv to standardize input
    - Classification branch: 2× 3×3 conv → class prediction
    - Regression branch: 2× 3×3 conv → bbox + objectness prediction
    
    Gen1 dataset has 2 classes: car, pedestrian
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 256,
        feat_channels: int = 256,
        strides: List[int] = [8, 16, 32]
    ):
        """
        Args:
            num_classes: Number of detection classes (2 for Gen1)
            in_channels: Input feature channels from backbone
            feat_channels: Internal feature channels
            strides: Strides for each scale level
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = strides
        self.n_scales = len(strides)
        
        # Create branches for each scale
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        
        for _ in range(self.n_scales):
            # Stem conv
            self.stems.append(
                ConvBNReLU(in_channels, feat_channels, kernel_size=1, padding=0)
            )
            
            # Classification branch
            self.cls_convs.append(nn.Sequential(
                ConvBNReLU(feat_channels, feat_channels, kernel_size=3, padding=1),
                ConvBNReLU(feat_channels, feat_channels, kernel_size=3, padding=1)
            ))
            
            # Regression branch
            self.reg_convs.append(nn.Sequential(
                ConvBNReLU(feat_channels, feat_channels, kernel_size=3, padding=1),
                ConvBNReLU(feat_channels, feat_channels, kernel_size=3, padding=1)
            ))
            
            # Prediction layers
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, kernel_size=1))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, kernel_size=1))  # x, y, w, h
            self.obj_preds.append(nn.Conv2d(feat_channels, 1, kernel_size=1))  # objectness
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize prediction layers with proper biases."""
        for cls_pred in self.cls_preds:
            # Initialize classification bias for better training stability
            nn.init.constant_(cls_pred.bias, -4.6)  # -log((1-0.01)/0.01)
        
        for obj_pred in self.obj_preds:
            # Initialize objectness bias
            nn.init.constant_(obj_pred.bias, -4.6)
    
    def forward(
        self, 
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
            features: Multi-scale feature dict {'p3', 'p4', 'p5'}
        
        Returns:
            outputs: Detection outputs per scale
                Each scale contains:
                - 'cls': [B, num_classes, H, W] class logits
                - 'reg': [B, 4, H, W] bbox regression
                - 'obj': [B, 1, H, W] objectness logits
        """
        outputs = {}
        scale_names = ['p3', 'p4', 'p5']
        
        for idx, scale_name in enumerate(scale_names):
            feat = features[scale_name]
            
            # Stem
            x = self.stems[idx](feat)
            
            # Classification branch
            cls_feat = self.cls_convs[idx](x)
            cls_pred = self.cls_preds[idx](cls_feat)
            
            # Regression branch
            reg_feat = self.reg_convs[idx](x)
            reg_pred = self.reg_preds[idx](reg_feat)
            obj_pred = self.obj_preds[idx](reg_feat)
            
            outputs[scale_name] = {
                'cls': cls_pred,  # [B, num_classes, H, W]
                'reg': reg_pred,  # [B, 4, H, W]
                'obj': obj_pred   # [B, 1, H, W]
            }
        
        return outputs
    
    def decode_outputs(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode outputs to bounding boxes in original image coordinates.
        
        Args:
            outputs: Raw model outputs
            image_size: (H, W) of input image
        
        Returns:
            boxes: [N, 4] in (x1, y1, x2, y2) format
            scores: [N] confidence scores
            labels: [N] class labels
        """
        device = outputs['p3']['cls'].device
        batch_size = outputs['p3']['cls'].shape[0]
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for scale_idx, scale_name in enumerate(['p3', 'p4', 'p5']):
            stride = self.strides[scale_idx]
            
            cls_pred = outputs[scale_name]['cls']  # [B, num_classes, H, W]
            reg_pred = outputs[scale_name]['reg']  # [B, 4, H, W]
            obj_pred = outputs[scale_name]['obj']  # [B, 1, H, W]
            
            _, _, H, W = cls_pred.shape
            
            # Create grid
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            grid = torch.stack([xv, yv], dim=0).float()  # [2, H, W]
            
            # Decode bbox
            # reg_pred: [dx, dy, dw, dh] where d* are offsets from grid cell
            reg_pred = reg_pred.sigmoid()  # Normalize to [0, 1]
            
            for b in range(batch_size):
                # Objectness scores
                obj_scores = obj_pred[b].sigmoid().squeeze(0)  # [H, W]
                
                # Class scores
                cls_scores = cls_pred[b].sigmoid()  # [num_classes, H, W]
                
                # Combined scores: objectness * class_score
                # [num_classes, H, W]
                scores = obj_scores.unsqueeze(0) * cls_scores
                
                # Find positions above threshold
                max_scores, labels = scores.max(dim=0)  # [H, W]
                mask = max_scores > 0.01
                
                if mask.sum() > 0:
                    ys, xs = torch.where(mask)
                    
                    for y, x in zip(ys, xs):
                        # Get regression values
                        dx, dy, dw, dh = reg_pred[b, :, y, x]
                        
                        # Decode to image coordinates
                        cx = (x.float() + dx) * stride
                        cy = (y.float() + dy) * stride
                        w = dw * stride * 4  # Scale factor
                        h = dh * stride * 4
                        
                        # Convert to x1y1x2y2
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        
                        all_boxes.append(torch.tensor([x1, y1, x2, y2], device=device))
                        all_scores.append(max_scores[y, x])
                        all_labels.append(labels[y, x])
        
        if len(all_boxes) > 0:
            boxes = torch.stack(all_boxes)
            scores = torch.stack(all_scores)
            labels = torch.stack(all_labels)
        else:
            boxes = torch.zeros((0, 4), device=device)
            scores = torch.zeros((0,), device=device)
            labels = torch.zeros((0,), dtype=torch.long, device=device)
        
        return boxes, scores, labels
    
    def get_param_count(self) -> int:
        """Returns parameter count for this component."""
        return sum(p.numel() for p in self.parameters())


# Testing code
if __name__ == '__main__':
    print("Testing YOLOX Head Implementation...")
    
    # Create head
    head = YOLOXHead(num_classes=2, in_channels=256)
    
    # Multi-scale features (from ANN backbone)
    # For Gen1 (304×240 input): p3=38×30, p4=19×15, p5=9×7
    B = 2
    features = {
        'p3': torch.randn(B, 256, 38, 30),
        'p4': torch.randn(B, 256, 19, 15),
        'p5': torch.randn(B, 256, 9, 7)
    }
    
    print("Input feature shapes:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = head(features)
    
    print("\nOutput shapes:")
    for scale_name, scale_out in outputs.items():
        print(f"  {scale_name}:")
        for key, val in scale_out.items():
            print(f"    {key}: {val.shape}")
    
    # Verify output dimensions
    for scale_name in ['p3', 'p4', 'p5']:
        assert 'cls' in outputs[scale_name]
        assert 'reg' in outputs[scale_name]
        assert 'obj' in outputs[scale_name]
        assert outputs[scale_name]['cls'].shape[1] == 2, "cls channels should be num_classes"
        assert outputs[scale_name]['reg'].shape[1] == 4, "reg channels should be 4"
        assert outputs[scale_name]['obj'].shape[1] == 1, "obj channels should be 1"
    
    # Parameter count
    param_count = head.get_param_count()
    print(f"\nYOLOX Head Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"Expected: ~2.8M (per paper)")
    
    print("\n✓ YOLOX Head tests passed!")
