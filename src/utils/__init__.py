"""
Utility modules for training and evaluation.

Components:
- YOLOXLoss: Detection loss function
- compute_map: mAP metric computation
- bbox_iou: IoU computation
"""

from .losses import YOLOXLoss, bbox_iou
from .metrics import compute_map, compute_ap

__all__ = [
    'YOLOXLoss',
    'bbox_iou',
    'compute_map',
    'compute_ap',
]
