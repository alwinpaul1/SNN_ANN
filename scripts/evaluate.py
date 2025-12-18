"""
Evaluation Script for Hybrid SNN-ANN Detector

Evaluates a trained model on the Gen1 test set and reports mAP metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pth --data_root ./data/gen1

Paper expected results (Table 1-2):
- mAP(.5:.05:.95): 0.35
- mAP(.5): 0.61
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hybrid_model import build_model
from src.data.gen1_dataset import Gen1Dataset, collate_fn
from src.utils.metrics import MAPEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Hybrid SNN-ANN Detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data/gen1',
                        help='Path to Gen1 dataset')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--num_time_bins', type=int, default=10,
                        help='Number of time bins')
    parser.add_argument('--score_thresh', type=float, default=0.01,
                        help='Score threshold for predictions')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = build_model(
        num_classes=args.num_classes,
        num_time_bins=args.num_time_bins,
        pretrained=True,
        checkpoint_path=args.checkpoint
    ).to(device)
    model.eval()
    
    print(f"Model parameters: {model.get_param_count():,}")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset from: {args.data_root}")
    dataset = Gen1Dataset(
        root_dir=args.data_root,
        split=args.split,
        num_time_bins=args.num_time_bins,
        augment=False
    )
    
    print(f"Samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No samples found! Check the data path.")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Evaluate
    print("\nEvaluating...")
    evaluator = MAPEvaluator(num_classes=args.num_classes)
    use_amp = not args.no_amp and torch.cuda.is_available()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            events = batch['events'].to(device)
            labels = batch['labels']
            
            if use_amp:
                with autocast():
                    predictions = model.predict(events, score_thresh=args.score_thresh)
            else:
                predictions = model.predict(events, score_thresh=args.score_thresh)
            
            evaluator.update(predictions, labels)
    
    # Compute metrics
    metrics = evaluator.compute()
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nmAP@0.5:       {metrics['mAP_50']:.4f}")
    print(f"mAP@0.5:0.95:  {metrics['mAP']:.4f}")
    
    # Per-class results
    print("\nPer-class AP@0.5:")
    for key, value in metrics.items():
        if key.startswith('AP_50_'):
            class_name = key.replace('AP_50_', '')
            print(f"  {class_name}: {value:.4f}")
    
    print("\n" + "="*50)
    print("PAPER EXPECTED RESULTS")
    print("="*50)
    print(f"mAP@0.5:       0.61")
    print(f"mAP@0.5:0.95:  0.35")
    print("="*50)


if __name__ == '__main__':
    main()
