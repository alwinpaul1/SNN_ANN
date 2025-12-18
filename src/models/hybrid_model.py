"""
Complete Hybrid SNN-ANN Detector

Architecture Flow (from paper Figure 1):
Events → To Tensor → SNN Blocks → β_asab → ANN Blocks → Detection FPN + Head → Detections

This implements the base Hybrid model (without DWConvLSTM).
For +RNN variant, add DWConvLSTM between ANN blocks.

Paper Reference: Ahmed et al. (arXiv:2403.10173v4)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .snn_backbone import SNNBackbone
from .asab_module import ASABModule
from .ann_backbone import ANNBackbone
from .yolox_head import YOLOXHead


class HybridBackbone(nn.Module):
    """
    Hybrid SNN-ANN Backbone
    
    Combines:
    - SNN: Low-level spatiotemporal feature extraction
    - ASAB: Attention-based bridge (sparse→dense conversion)
    - ANN: High-level spatial feature extraction
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        tau_init: float = 2.0,
        asab_kernel_size: int = 5,
        num_time_bins: int = 10
    ):
        super().__init__()
        
        # SNN Backbone: extracts low-level spatiotemporal features
        self.snn = SNNBackbone(in_channels=in_channels, tau_init=tau_init)
        
        # ASAB Bridge: converts sparse spikes to dense features
        self.asab = ASABModule(
            channels=256,
            kernel_size=asab_kernel_size,
            num_time_bins=num_time_bins
        )
        
        # ANN Backbone: extracts high-level spatial features
        self.ann = ANNBackbone(in_channels=256)
    
    def forward(self, event_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            event_tensor: [T, B, 2, H, W] - Event representation
        
        Returns:
            features: Multi-scale feature dict for detection head
        """
        # SNN: Extract spatiotemporal features
        # [T, B, 2, H, W] → [T, B, 256, H/8, W/8]
        E_spike = self.snn(event_tensor)
        
        # ASAB: Bridge sparse→dense
        # [T, B, 256, H/8, W/8] → [B, 256, H/8, W/8]
        F_out = self.asab(E_spike)
        
        # ANN: Extract high-level features
        # [B, 256, H/8, W/8] → multi-scale features
        features = self.ann(F_out)
        
        return features


class HybridSNNANNDetector(nn.Module):
    """
    Complete Hybrid SNN-ANN Object Detector
    
    For Gen1 dataset:
    - Input: Event tensor [T, B, 2, 304, 240]
    - Output: Detections (bboxes, classes, scores)
    - Classes: 2 (car, pedestrian)
    
    Expected Performance (from paper):
    - mAP(.5:.05:.95): 0.35
    - mAP(.5): 0.61
    - Parameters: ~6.6M
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # CORRECTED: Gen1 has 2 classes
        in_channels: int = 2,
        tau_init: float = 2.0,
        asab_kernel_size: int = 5,
        num_time_bins: int = 10
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_time_bins = num_time_bins
        
        # Hybrid backbone
        self.backbone = HybridBackbone(
            in_channels=in_channels,
            tau_init=tau_init,
            asab_kernel_size=asab_kernel_size,
            num_time_bins=num_time_bins
        )
        
        # Detection head
        self.head = YOLOXHead(num_classes=num_classes, in_channels=256)
    
    def forward(
        self,
        event_tensor: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
            event_tensor: [T, B, 2, H, W]
            targets: Optional list of target dicts for training
        
        Returns:
            outputs: Detection outputs per scale
        """
        # Extract features
        features = self.backbone(event_tensor)
        
        # Detection head
        outputs = self.head(features)
        
        return outputs
    
    @torch.no_grad()
    def predict(
        self,
        event_tensor: torch.Tensor,
        score_thresh: float = 0.3,
        nms_thresh: float = 0.5
    ) -> List[Dict]:
        """
        Inference with NMS post-processing.
        
        Args:
            event_tensor: [T, B, 2, H, W]
            score_thresh: Score threshold for filtering
            nms_thresh: IoU threshold for NMS
        
        Returns:
            predictions: List of dicts with 'boxes', 'scores', 'labels'
        """
        self.eval()
        
        outputs = self.forward(event_tensor)
        
        batch_size = event_tensor.shape[1]
        predictions = []
        
        for b in range(batch_size):
            # Collect predictions from all scales
            all_boxes = []
            all_scores = []
            all_labels = []
            
            for scale_idx, scale_name in enumerate(['p3', 'p4', 'p5']):
                stride = self.head.strides[scale_idx]
                
                cls_pred = outputs[scale_name]['cls'][b]  # [num_classes, H, W]
                reg_pred = outputs[scale_name]['reg'][b]  # [4, H, W]
                obj_pred = outputs[scale_name]['obj'][b]  # [1, H, W]
                
                _, H, W = cls_pred.shape
                device = cls_pred.device
                
                # Create grid
                yv, xv = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing='ij'
                )
                
                # Decode predictions
                obj_scores = obj_pred.sigmoid().squeeze(0)  # [H, W]
                cls_scores = cls_pred.sigmoid()  # [num_classes, H, W]
                
                # Combined score
                scores = obj_scores.unsqueeze(0) * cls_scores
                max_scores, labels = scores.max(dim=0)
                
                # Filter by score
                mask = max_scores > score_thresh
                
                if mask.sum() > 0:
                    ys, xs = torch.where(mask)
                    
                    for i in range(len(ys)):
                        y, x = ys[i], xs[i]
                        
                        # Decode bbox
                        dx, dy, dw, dh = reg_pred[:, y, x]
                        
                        # YOLOX decoding
                        cx = (x.float() + dx.sigmoid()) * stride
                        cy = (y.float() + dy.sigmoid()) * stride
                        w = torch.exp(dw) * stride
                        h = torch.exp(dh) * stride
                        
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
                
                # Apply NMS per class
                keep_boxes = []
                keep_scores = []
                keep_labels = []
                
                for cls_id in range(self.num_classes):
                    cls_mask = labels == cls_id
                    if cls_mask.sum() > 0:
                        cls_boxes = boxes[cls_mask]
                        cls_scores = scores[cls_mask]
                        
                        # Simple NMS
                        keep = self._nms(cls_boxes, cls_scores, nms_thresh)
                        
                        keep_boxes.append(cls_boxes[keep])
                        keep_scores.append(cls_scores[keep])
                        keep_labels.append(torch.full((len(keep),), cls_id, device=device))
                
                if len(keep_boxes) > 0:
                    boxes = torch.cat(keep_boxes)
                    scores = torch.cat(keep_scores)
                    labels = torch.cat(keep_labels)
                else:
                    boxes = torch.zeros((0, 4), device=device)
                    scores = torch.zeros((0,), device=device)
                    labels = torch.zeros((0,), dtype=torch.long, device=device)
            else:
                device = event_tensor.device
                boxes = torch.zeros((0, 4), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)
            
            predictions.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        
        return predictions
    
    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_thresh: float
    ) -> torch.Tensor:
        """Simple NMS implementation."""
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # Sort by scores
        _, order = scores.sort(descending=True)
        
        keep = []
        while len(order) > 0:
            i = order[0].item()
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            remaining = order[1:]
            ious = self._box_iou(boxes[i:i+1], boxes[remaining])[0]
            
            # Keep boxes with IoU below threshold
            mask = ious < iou_thresh
            order = remaining[mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = area1[:, None] + area2 - inter_area
        
        return inter_area / (union_area + 1e-7)
    
    def get_param_count(self) -> int:
        """Returns total parameter count (paper reports 6.6M)."""
        return sum(p.numel() for p in self.parameters())
    
    def get_component_params(self) -> Dict[str, int]:
        """Returns parameter count per component."""
        return {
            'snn_backbone': self.backbone.snn.get_param_count(),
            'asab': self.backbone.asab.get_param_count(),
            'ann_backbone': self.backbone.ann.get_param_count(),
            'yolox_head': self.head.get_param_count(),
        }


def build_model(
    num_classes: int = 2,
    num_time_bins: int = 10,
    tau_init: float = 2.0,
    asab_kernel_size: int = 5,
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None
) -> HybridSNNANNDetector:
    """
    Factory function to build the Hybrid SNN-ANN detector.
    
    Args:
        num_classes: Number of detection classes (2 for Gen1)
        num_time_bins: Number of time bins T (10 for Gen1)
        tau_init: Initial PLIF time constant
        asab_kernel_size: TSDC kernel size (5 per paper)
        pretrained: Load pretrained weights
        checkpoint_path: Path to checkpoint file
    
    Returns:
        model: HybridSNNANNDetector instance
    """
    model = HybridSNNANNDetector(
        num_classes=num_classes,
        in_channels=2,
        tau_init=tau_init,
        asab_kernel_size=asab_kernel_size,
        num_time_bins=num_time_bins
    )
    
    if pretrained and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    return model


# Testing code
if __name__ == '__main__':
    print("Testing Hybrid SNN-ANN Detector Implementation...")
    print("="*60)
    
    # Create model
    model = HybridSNNANNDetector(
        num_classes=2,
        in_channels=2,
        tau_init=2.0,
        asab_kernel_size=5,
        num_time_bins=10
    )
    
    # Gen1 input dimensions: T=10, B=2, C=2, H=304, W=240
    T, B, C, H, W = 10, 2, 2, 304, 240
    event_tensor = torch.randn(T, B, C, H, W)
    
    print(f"\nInput shape: {event_tensor.shape}")
    print(f"  T={T} time bins")
    print(f"  B={B} batch size")
    print(f"  C={C} polarity channels")
    print(f"  H×W={H}×{W} spatial dimensions")
    
    # Forward pass
    print("\n" + "-"*40)
    print("Forward pass...")
    with torch.no_grad():
        outputs = model(event_tensor)
    
    print("\nOutput shapes:")
    for scale_name, scale_out in outputs.items():
        print(f"  {scale_name}:")
        for key, val in scale_out.items():
            print(f"    {key}: {val.shape}")
    
    # Parameter count
    print("\n" + "-"*40)
    print("Parameter count:")
    
    total_params = model.get_param_count()
    component_params = model.get_component_params()
    
    for name, count in component_params.items():
        print(f"  {name}: {count:,} ({count/1e6:.2f}M)")
    
    print(f"\n  TOTAL: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Expected (paper): ~6.6M")
    
    # Test inference
    print("\n" + "-"*40)
    print("Testing inference...")
    
    predictions = model.predict(event_tensor, score_thresh=0.1)
    
    print(f"\nPredictions for batch of {B}:")
    for i, pred in enumerate(predictions):
        print(f"  Image {i}:")
        print(f"    Boxes: {pred['boxes'].shape}")
        print(f"    Scores: {pred['scores'].shape}")
        print(f"    Labels: {pred['labels'].shape}")
    
    # Test gradient flow
    print("\n" + "-"*40)
    print("Testing gradient flow...")
    
    event_tensor_grad = torch.randn(T, B, C, H, W, requires_grad=True)
    outputs_grad = model(event_tensor_grad)
    
    # Compute dummy loss
    loss = sum(
        out['cls'].sum() + out['reg'].sum() + out['obj'].sum()
        for out in outputs_grad.values()
    )
    loss.backward()
    
    print(f"Input gradient exists: {event_tensor_grad.grad is not None}")
    print(f"Input gradient shape: {event_tensor_grad.grad.shape}")
    
    print("\n" + "="*60)
    print("✓ All Hybrid SNN-ANN Detector tests passed!")
