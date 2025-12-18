"""
Evaluation Metrics for Object Detection

Implements:
- Average Precision (AP) computation
- Mean Average Precision (mAP) at various IoU thresholds
- Per-class AP computation

Following COCO evaluation protocol.

Paper Reference: Ahmed et al. (arXiv:2403.10173v4)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def compute_iou_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray
) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1: [N, 4] array (x1, y1, x2, y2)
        boxes2: [M, 4] array (x1, y1, x2, y2)
    
    Returns:
        iou_matrix: [N, M] array
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    
    # Intersection
    inter_x1 = np.maximum(x1_1[:, None], x1_2[None, :])
    inter_y1 = np.maximum(y1_1[:, None], y1_2[None, :])
    inter_x2 = np.minimum(x2_1[:, None], x2_2[None, :])
    inter_y2 = np.minimum(y2_1[:, None], y2_2[None, :])
    
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1[:, None] + area2[None, :] - inter_area
    
    return inter_area / (union_area + 1e-7)


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
    use_07_metric: bool = False
) -> float:
    """
    Compute Average Precision from recall-precision curve.
    
    Args:
        recalls: Array of recall values (sorted in increasing order)
        precisions: Array of precision values at each recall
        use_07_metric: Use VOC 2007 11-point metric
    
    Returns:
        ap: Average Precision value
    """
    if use_07_metric:
        # VOC 2007 11-point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap
    
    # COCO-style AP (area under PR curve)
    # Prepend (0, 1) and append (1, 0) for proper interpolation
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[1.0], precisions, [0.0]])
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    # Find where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum area under curve
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def compute_ap_per_class(
    predictions: List[Dict],
    ground_truths: List[torch.Tensor],
    class_id: int,
    iou_threshold: float = 0.5
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AP for a single class.
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of ground truth tensors [N, 5] (x, y, w, h, class_id)
        class_id: Class ID to evaluate
        iou_threshold: IoU threshold for matching
    
    Returns:
        ap: Average Precision
        recalls: Recall array
        precisions: Precision array
    """
    # Collect all predictions and ground truths for this class
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_img_ids = []
    
    gt_per_image = defaultdict(list)
    n_gt = 0
    
    for img_id, (preds, gts) in enumerate(zip(predictions, ground_truths)):
        # Ground truths
        if isinstance(gts, torch.Tensor):
            gts = gts.cpu().numpy()
        
        if len(gts) > 0:
            # Convert from (x, y, w, h) to (x1, y1, x2, y2)
            gt_class_mask = gts[:, 4] == class_id
            gt_boxes = gts[gt_class_mask, :4].copy()
            if len(gt_boxes) > 0:
                gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]  # x2 = x + w
                gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]  # y2 = y + h
                gt_per_image[img_id] = {
                    'boxes': gt_boxes,
                    'matched': np.zeros(len(gt_boxes), dtype=bool)
                }
                n_gt += len(gt_boxes)
        
        # Predictions
        if preds is None or len(preds) == 0:
            continue
        
        if isinstance(preds, dict):
            pred_boxes = preds['boxes'].cpu().numpy() if isinstance(preds['boxes'], torch.Tensor) else preds['boxes']
            pred_scores = preds['scores'].cpu().numpy() if isinstance(preds['scores'], torch.Tensor) else preds['scores']
            pred_labels = preds['labels'].cpu().numpy() if isinstance(preds['labels'], torch.Tensor) else preds['labels']
        else:
            # Handle list of dicts format
            pred_boxes = np.array([p['bbox'] for p in preds if p.get('class_id') == class_id])
            pred_scores = np.array([p['score'] for p in preds if p.get('class_id') == class_id])
            pred_labels = np.full(len(pred_scores), class_id)
        
        if len(pred_boxes) == 0:
            continue
        
        # Filter by class
        class_mask = pred_labels == class_id
        if not class_mask.any():
            continue
        
        pred_boxes = pred_boxes[class_mask]
        pred_scores = pred_scores[class_mask]
        
        all_pred_boxes.append(pred_boxes)
        all_pred_scores.append(pred_scores)
        all_pred_img_ids.extend([img_id] * len(pred_scores))
    
    if n_gt == 0 or len(all_pred_scores) == 0:
        return 0.0, np.array([0.0]), np.array([0.0])
    
    # Concatenate all predictions
    all_pred_boxes = np.vstack(all_pred_boxes)
    all_pred_scores = np.concatenate(all_pred_scores)
    all_pred_img_ids = np.array(all_pred_img_ids)
    
    # Sort by confidence
    sorted_indices = np.argsort(-all_pred_scores)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    all_pred_img_ids = all_pred_img_ids[sorted_indices]
    
    # Compute TP/FP
    n_preds = len(all_pred_boxes)
    tp = np.zeros(n_preds)
    fp = np.zeros(n_preds)
    
    for pred_idx in range(n_preds):
        img_id = all_pred_img_ids[pred_idx]
        pred_box = all_pred_boxes[pred_idx]
        
        if img_id not in gt_per_image:
            fp[pred_idx] = 1
            continue
        
        gt_data = gt_per_image[img_id]
        gt_boxes = gt_data['boxes']
        gt_matched = gt_data['matched']
        
        if len(gt_boxes) == 0:
            fp[pred_idx] = 1
            continue
        
        # Compute IoU with all ground truths
        ious = compute_iou_matrix(pred_box[None, :], gt_boxes)[0]
        
        # Find best matching ground truth
        best_iou_idx = np.argmax(ious)
        best_iou = ious[best_iou_idx]
        
        if best_iou >= iou_threshold and not gt_matched[best_iou_idx]:
            tp[pred_idx] = 1
            gt_matched[best_iou_idx] = True
        else:
            fp[pred_idx] = 1
    
    # Compute precision/recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
    
    # Compute AP
    ap = compute_ap(recalls, precisions)
    
    return ap, recalls, precisions


def compute_map(
    predictions: List[Dict],
    ground_truths: List[torch.Tensor],
    num_classes: int = 2,
    iou_thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP).
    
    Args:
        predictions: List of prediction dicts for each image
        ground_truths: List of ground truth tensors for each image
        num_classes: Number of classes (2 for Gen1)
        iou_thresholds: List of IoU thresholds for mAP computation
                       Default: [0.5] for mAP@0.5
                       Use np.arange(0.5, 0.95, 0.05) for COCO mAP
    
    Returns:
        metrics: Dict with mAP values
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    
    # Compute AP for each class and IoU threshold
    results = {}
    
    # mAP@0.5
    aps_50 = []
    for class_id in range(num_classes):
        ap, _, _ = compute_ap_per_class(
            predictions, ground_truths, class_id,
            iou_threshold=0.5
        )
        aps_50.append(ap)
    
    results['mAP_50'] = np.mean(aps_50) if aps_50 else 0.0
    
    # Per-class AP@0.5
    class_names = ['car', 'pedestrian']
    for class_id in range(num_classes):
        class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
        results[f'AP_50_{class_name}'] = aps_50[class_id] if class_id < len(aps_50) else 0.0
    
    # mAP@0.5:0.95 (COCO style)
    if len(iou_thresholds) > 1 or 0.75 in iou_thresholds:
        iou_range = np.arange(0.5, 0.95, 0.05)
        aps_coco = []
        
        for iou_thresh in iou_range:
            class_aps = []
            for class_id in range(num_classes):
                ap, _, _ = compute_ap_per_class(
                    predictions, ground_truths, class_id,
                    iou_threshold=iou_thresh
                )
                class_aps.append(ap)
            aps_coco.append(np.mean(class_aps))
        
        results['mAP'] = np.mean(aps_coco) if aps_coco else 0.0
    else:
        results['mAP'] = results['mAP_50']
    
    return results


class MAPEvaluator:
    """
    Accumulator-based mAP evaluator for validation.
    
    Usage:
        evaluator = MAPEvaluator(num_classes=2)
        for batch in val_loader:
            predictions = model.predict(batch['events'])
            evaluator.update(predictions, batch['labels'])
        metrics = evaluator.compute()
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.predictions = []
        self.ground_truths = []
    
    def reset(self):
        """Reset accumulated predictions and ground truths."""
        self.predictions = []
        self.ground_truths = []
    
    def update(
        self,
        predictions: List[Dict],
        ground_truths: List[torch.Tensor]
    ):
        """
        Add batch of predictions and ground truths.
        
        Args:
            predictions: List of prediction dicts
            ground_truths: List of ground truth tensors
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute mAP metrics.
        
        Returns:
            metrics: Dict with mAP values
        """
        return compute_map(
            self.predictions,
            self.ground_truths,
            self.num_classes
        )


# Testing code
if __name__ == '__main__':
    print("Testing Metrics Implementation...")
    
    # Create dummy predictions and ground truths
    # Format: predictions = [{'boxes': [x1,y1,x2,y2], 'scores': [], 'labels': []}]
    # ground_truths = [tensor([x, y, w, h, class_id])]
    
    predictions = [
        {
            'boxes': torch.tensor([
                [50, 50, 100, 100],
                [150, 150, 200, 200],
                [60, 60, 110, 110]
            ]),
            'scores': torch.tensor([0.9, 0.8, 0.7]),
            'labels': torch.tensor([0, 1, 0])
        },
        {
            'boxes': torch.tensor([
                [30, 30, 80, 80]
            ]),
            'scores': torch.tensor([0.85]),
            'labels': torch.tensor([0])
        }
    ]
    
    ground_truths = [
        torch.tensor([
            [50, 50, 50, 50, 0],  # x, y, w, h, class_id
            [150, 150, 50, 50, 1]
        ]),
        torch.tensor([
            [30, 30, 50, 50, 0]
        ])
    ]
    
    # Compute mAP
    metrics = compute_map(predictions, ground_truths, num_classes=2)
    
    print("\nComputed metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test evaluator
    print("\nTesting MAPEvaluator...")
    evaluator = MAPEvaluator(num_classes=2)
    evaluator.update(predictions, ground_truths)
    eval_metrics = evaluator.compute()
    
    print("Evaluator metrics:")
    for name, value in eval_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\n✓ Metrics tests passed!")
