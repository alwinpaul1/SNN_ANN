"""
YOLOX Loss Functions

Paper uses YOLOX framework with:
- IoU loss for bounding box regression
- Class loss (BCE)
- Objectness loss (BCE)

Reference: 
- Ahmed et al. (arXiv:2403.10173v4)
- YOLOX: Ge et al., 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    x1y1x2y2: bool = False,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate IoU (and variants) between boxes.
    
    Args:
        box1: [N, 4] boxes
        box2: [M, 4] boxes
        x1y1x2y2: If True, boxes are (x1, y1, x2, y2), else (cx, cy, w, h)
        GIoU: Return Generalized IoU
        DIoU: Return Distance IoU
        CIoU: Return Complete IoU
        eps: Small value to avoid division by zero
    
    Returns:
        iou: [N, M] IoU matrix (or scalar if N=M=1)
    """
    if not x1y1x2y2:
        # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
        
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    
    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + eps
    
    iou = inter_area / union_area
    
    if GIoU or DIoU or CIoU:
        # Enclosing box
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            
            if DIoU:
                return iou - rho2 / c2
            
            elif CIoU:
                v = (4 / (3.14159 ** 2)) * torch.pow(
                    torch.atan((b2_x2 - b2_x1) / (b2_y2 - b2_y1 + eps)) -
                    torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1 + eps)), 2
                )
                with torch.no_grad():
                    alpha = v / (1 - iou + v + eps)
                return iou - (rho2 / c2 + v * alpha)
        
        else:  # GIoU
            c_area = cw * ch + eps
            return iou - (c_area - union_area) / c_area
    
    return iou


def bbox_iou_matrix(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    x1y1x2y2: bool = True,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1: [N, 4] boxes
        boxes2: [M, 4] boxes
        x1y1x2y2: If True, boxes are (x1, y1, x2, y2)
        eps: Small value to avoid division by zero
    
    Returns:
        iou_matrix: [N, M] IoU values
    """
    if not x1y1x2y2:
        # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
        boxes1 = torch.cat([
            boxes1[:, :2] - boxes1[:, 2:] / 2,
            boxes1[:, :2] + boxes1[:, 2:] / 2
        ], dim=1)
        boxes2 = torch.cat([
            boxes2[:, :2] - boxes2[:, 2:] / 2,
            boxes2[:, :2] + boxes2[:, 2:] / 2
        ], dim=1)
    
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # Expand for broadcasting
    b1 = boxes1.unsqueeze(1).expand(N, M, 4)
    b2 = boxes2.unsqueeze(0).expand(N, M, 4)
    
    # Intersection
    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union_area = area1 + area2 - inter_area + eps
    
    return inter_area / union_area


class YOLOXLoss(nn.Module):
    """
    YOLOX Loss combining:
    - IoU loss for regression
    - BCE loss for objectness
    - BCE loss for classification
    
    With SimOTA-style label assignment.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        strides: List[int] = [8, 16, 32],
        reg_weight: float = 5.0,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        use_l1: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = strides
        self.reg_weight = reg_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.use_l1 = use_l1
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def forward(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        targets: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            outputs: Detection outputs per scale
                Each scale has 'cls', 'reg', 'obj'
            targets: List of [N_i, 5] target boxes per image
                Format: (x, y, w, h, class_id) - x,y is top-left corner
            image_sizes: List of (H, W) tuples
        
        Returns:
            losses: Dict with 'total', 'reg', 'obj', 'cls'
        """
        device = outputs['p3']['cls'].device
        batch_size = outputs['p3']['cls'].shape[0]
        
        total_reg_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_l1_loss = torch.tensor(0.0, device=device)
        
        num_pos = 0
        
        # Process each image in batch
        for b in range(batch_size):
            target = targets[b]  # [N, 5]: x, y, w, h, class_id
            
            # Collect predictions from all scales
            all_cls_preds = []
            all_reg_preds = []
            all_obj_preds = []
            all_grids = []
            all_strides = []
            
            for scale_idx, scale_name in enumerate(['p3', 'p4', 'p5']):
                stride = self.strides[scale_idx]
                
                cls_pred = outputs[scale_name]['cls'][b]  # [num_classes, H, W]
                reg_pred = outputs[scale_name]['reg'][b]  # [4, H, W]
                obj_pred = outputs[scale_name]['obj'][b]  # [1, H, W]
                
                _, H, W = cls_pred.shape
                
                # Create grid
                yv, xv = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing='ij'
                )
                grid = torch.stack([xv, yv], dim=-1).float().reshape(-1, 2)  # [H*W, 2]
                
                # Flatten predictions
                cls_pred_flat = cls_pred.permute(1, 2, 0).reshape(-1, self.num_classes)
                reg_pred_flat = reg_pred.permute(1, 2, 0).reshape(-1, 4)
                obj_pred_flat = obj_pred.permute(1, 2, 0).reshape(-1, 1)
                
                all_cls_preds.append(cls_pred_flat)
                all_reg_preds.append(reg_pred_flat)
                all_obj_preds.append(obj_pred_flat)
                all_grids.append(grid)
                all_strides.append(torch.full((H * W,), stride, device=device))
            
            # Concatenate all scales
            cls_preds = torch.cat(all_cls_preds, dim=0)  # [total_anchors, num_classes]
            reg_preds = torch.cat(all_reg_preds, dim=0)  # [total_anchors, 4]
            obj_preds = torch.cat(all_obj_preds, dim=0)  # [total_anchors, 1]
            grids = torch.cat(all_grids, dim=0)          # [total_anchors, 2]
            strides = torch.cat(all_strides, dim=0)      # [total_anchors]
            
            total_anchors = cls_preds.shape[0]
            
            if len(target) == 0:
                # No targets - only objectness loss
                obj_targets = torch.zeros_like(obj_preds)
                total_obj_loss += self.bce_loss(obj_preds, obj_targets).sum()
                continue
            
            # Decode predicted boxes to image coordinates
            # pred_boxes: [total_anchors, 4] in (cx, cy, w, h)
            pred_cx = (grids[:, 0] + reg_preds[:, 0].sigmoid()) * strides
            pred_cy = (grids[:, 1] + reg_preds[:, 1].sigmoid()) * strides
            pred_w = torch.exp(reg_preds[:, 2]) * strides
            pred_h = torch.exp(reg_preds[:, 3]) * strides
            pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)
            
            # Convert targets to cx, cy, w, h format
            # targets are (x, y, w, h) where x, y is top-left
            target_cx = target[:, 0] + target[:, 2] / 2
            target_cy = target[:, 1] + target[:, 3] / 2
            target_w = target[:, 2]
            target_h = target[:, 3]
            target_boxes = torch.stack([target_cx, target_cy, target_w, target_h], dim=1)
            target_classes = target[:, 4].long()
            
            # Simple assignment: find best matching anchor for each target
            # Using center-based assignment
            n_targets = len(target)
            
            # For each target, find anchors whose grid cell contains the target center
            assigned_anchors = []
            assigned_targets = []
            
            for t_idx in range(n_targets):
                t_cx, t_cy = target_cx[t_idx], target_cy[t_idx]
                
                # Find best anchors based on center distance
                anchor_cx = grids[:, 0] * strides + strides / 2
                anchor_cy = grids[:, 1] * strides + strides / 2
                
                center_dist = torch.sqrt((anchor_cx - t_cx) ** 2 + (anchor_cy - t_cy) ** 2)
                
                # Select top-k closest anchors
                k = min(10, total_anchors)
                _, topk_indices = center_dist.topk(k, largest=False)
                
                # Calculate IoU with selected anchors
                selected_pred_boxes = pred_boxes[topk_indices]
                ious = bbox_iou(
                    selected_pred_boxes,
                    target_boxes[t_idx:t_idx+1].expand(k, -1),
                    x1y1x2y2=False
                )
                
                # Select anchors with IoU > 0.1 or at least the best one
                valid_mask = ious > 0.1
                if not valid_mask.any():
                    best_idx = ious.argmax()
                    valid_indices = topk_indices[best_idx:best_idx+1]
                else:
                    valid_indices = topk_indices[valid_mask]
                
                assigned_anchors.extend(valid_indices.tolist())
                assigned_targets.extend([t_idx] * len(valid_indices))
            
            if len(assigned_anchors) == 0:
                # No assignments - use objectness loss only
                obj_targets = torch.zeros_like(obj_preds)
                total_obj_loss += self.bce_loss(obj_preds, obj_targets).sum()
                continue
            
            assigned_anchors = torch.tensor(assigned_anchors, device=device)
            assigned_targets = torch.tensor(assigned_targets, device=device)
            
            num_pos += len(assigned_anchors)
            
            # Get predictions for assigned anchors
            pos_pred_boxes = pred_boxes[assigned_anchors]
            pos_cls_preds = cls_preds[assigned_anchors]
            pos_obj_preds = obj_preds[assigned_anchors]
            pos_reg_preds = reg_preds[assigned_anchors]
            
            # Get target values
            pos_target_boxes = target_boxes[assigned_targets]
            pos_target_classes = target_classes[assigned_targets]
            
            # IoU loss for regression
            ious = bbox_iou(pos_pred_boxes, pos_target_boxes, x1y1x2y2=False, CIoU=True)
            reg_loss = (1.0 - ious).sum()
            total_reg_loss += reg_loss
            
            # L1 loss (optional)
            if self.use_l1:
                # Convert target boxes to regression targets
                pos_grids = grids[assigned_anchors]
                pos_strides = strides[assigned_anchors]
                
                target_dx = (pos_target_boxes[:, 0] / pos_strides - pos_grids[:, 0]).clamp(0, 1)
                target_dy = (pos_target_boxes[:, 1] / pos_strides - pos_grids[:, 1]).clamp(0, 1)
                target_dw = torch.log(pos_target_boxes[:, 2] / pos_strides + 1e-7)
                target_dh = torch.log(pos_target_boxes[:, 3] / pos_strides + 1e-7)
                
                target_regs = torch.stack([target_dx, target_dy, target_dw, target_dh], dim=1)
                pred_regs = torch.stack([
                    pos_reg_preds[:, 0].sigmoid(),
                    pos_reg_preds[:, 1].sigmoid(),
                    pos_reg_preds[:, 2],
                    pos_reg_preds[:, 3]
                ], dim=1)
                
                total_l1_loss += self.l1_loss(pred_regs, target_regs).sum()
            
            # Classification loss
            cls_targets = F.one_hot(pos_target_classes, self.num_classes).float()
            cls_loss = self.bce_loss(pos_cls_preds, cls_targets).sum()
            total_cls_loss += cls_loss
            
            # Objectness loss
            obj_targets = torch.zeros_like(obj_preds)
            obj_targets[assigned_anchors] = 1.0
            obj_loss = self.bce_loss(obj_preds, obj_targets).sum()
            total_obj_loss += obj_loss
        
        # Normalize by number of positive samples
        num_pos = max(num_pos, 1)
        
        total_reg_loss = total_reg_loss / num_pos
        total_obj_loss = total_obj_loss / (total_anchors * batch_size)  # Per anchor
        total_cls_loss = total_cls_loss / num_pos
        total_l1_loss = total_l1_loss / num_pos
        
        # Combine losses
        total_loss = (
            self.reg_weight * total_reg_loss +
            self.obj_weight * total_obj_loss +
            self.cls_weight * total_cls_loss
        )
        
        if self.use_l1:
            total_loss += total_l1_loss
        
        return {
            'total': total_loss,
            'reg': total_reg_loss,
            'obj': total_obj_loss,
            'cls': total_cls_loss,
            'l1': total_l1_loss if self.use_l1 else torch.tensor(0.0, device=device)
        }


# Testing code
if __name__ == '__main__':
    print("Testing YOLOX Loss Implementation...")
    
    # Create loss function
    loss_fn = YOLOXLoss(num_classes=2, strides=[8, 16, 32])
    
    # Create dummy outputs
    B = 2
    outputs = {
        'p3': {
            'cls': torch.randn(B, 2, 38, 30),
            'reg': torch.randn(B, 4, 38, 30),
            'obj': torch.randn(B, 1, 38, 30)
        },
        'p4': {
            'cls': torch.randn(B, 2, 19, 15),
            'reg': torch.randn(B, 4, 19, 15),
            'obj': torch.randn(B, 1, 19, 15)
        },
        'p5': {
            'cls': torch.randn(B, 2, 9, 7),
            'reg': torch.randn(B, 4, 9, 7),
            'obj': torch.randn(B, 1, 9, 7)
        }
    }
    
    # Create dummy targets (x, y, w, h, class_id)
    targets = [
        torch.tensor([[50, 50, 80, 60, 0], [100, 100, 50, 40, 1]], dtype=torch.float32),
        torch.tensor([[30, 40, 60, 80, 0]], dtype=torch.float32)
    ]
    
    image_sizes = [(304, 240), (304, 240)]
    
    # Compute loss
    losses = loss_fn(outputs, targets, image_sizes)
    
    print("\nLoss values:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    # Test with no targets
    print("\nTesting with no targets...")
    empty_targets = [torch.zeros((0, 5)), torch.zeros((0, 5))]
    losses_empty = loss_fn(outputs, empty_targets, image_sizes)
    print(f"  total: {losses_empty['total'].item():.4f}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    outputs_grad = {
        'p3': {
            'cls': torch.randn(B, 2, 38, 30, requires_grad=True),
            'reg': torch.randn(B, 4, 38, 30, requires_grad=True),
            'obj': torch.randn(B, 1, 38, 30, requires_grad=True)
        },
        'p4': {
            'cls': torch.randn(B, 2, 19, 15, requires_grad=True),
            'reg': torch.randn(B, 4, 19, 15, requires_grad=True),
            'obj': torch.randn(B, 1, 19, 15, requires_grad=True)
        },
        'p5': {
            'cls': torch.randn(B, 2, 9, 7, requires_grad=True),
            'reg': torch.randn(B, 4, 9, 7, requires_grad=True),
            'obj': torch.randn(B, 1, 9, 7, requires_grad=True)
        }
    }
    
    losses_grad = loss_fn(outputs_grad, targets, image_sizes)
    losses_grad['total'].backward()
    
    print(f"  cls grad exists: {outputs_grad['p3']['cls'].grad is not None}")
    print(f"  reg grad exists: {outputs_grad['p3']['reg'].grad is not None}")
    print(f"  obj grad exists: {outputs_grad['p3']['obj'].grad is not None}")
    
    print("\n✓ YOLOX Loss tests passed!")
