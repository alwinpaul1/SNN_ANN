"""
Training Script for Hybrid SNN-ANN Detector

Paper training setup (Section 4.1):
- Epochs: 50 for Gen1
- Batch size: 24 (paper used 4× 3090 GPUs)
- Learning rate: 2e-4
- Optimizer: Adam with OneCycle scheduler
- Training time: ~8 hours on 4× 3090 (expect ~4 hours on A100)

Paper Reference: Ahmed et al. (arXiv:2403.10173v4)

Usage:
    python src/train.py --config configs/gen1_config.yaml
    python src/train.py --data_root ./data/gen1 --epochs 50 --batch_size 8
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hybrid_model import HybridSNNANNDetector, build_model
from src.data.gen1_dataset import Gen1Dataset, collate_fn
from src.utils.losses import YOLOXLoss
from src.utils.metrics import MAPEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Hybrid SNN-ANN Detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file (overrides command line args)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data/gen1',
                        help='Path to Gen1 dataset root')
    parser.add_argument('--height', type=int, default=304,
                        help='Event frame height')
    parser.add_argument('--width', type=int, default=240,
                        help='Event frame width')
    parser.add_argument('--num_time_bins', type=int, default=10,
                        help='Number of time bins T')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of detection classes')
    parser.add_argument('--tau_init', type=float, default=2.0,
                        help='Initial PLIF time constant')
    parser.add_argument('--asab_kernel_size', type=int, default=5,
                        help='TSDC kernel size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--log_freq', type=int, default=50,
                        help='Log every N batches')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(args, config: dict):
    """Merge config file with command line arguments."""
    # Data config
    if 'data' in config:
        args.data_root = config['data'].get('root_dir', args.data_root)
        args.height = config['data'].get('height', args.height)
        args.width = config['data'].get('width', args.width)
        args.num_time_bins = config['data'].get('num_time_bins', args.num_time_bins)
    
    # Model config
    if 'model' in config:
        args.num_classes = config['model'].get('num_classes', args.num_classes)
        args.tau_init = config['model'].get('tau_init', args.tau_init)
        args.asab_kernel_size = config['model'].get('asab_kernel_size', args.asab_kernel_size)
    
    # Training config
    if 'training' in config:
        args.epochs = config['training'].get('epochs', args.epochs)
        args.batch_size = config['training'].get('batch_size', args.batch_size)
        args.learning_rate = config['training'].get('learning_rate', args.learning_rate)
        args.weight_decay = config['training'].get('weight_decay', args.weight_decay)
        args.num_workers = config['training'].get('num_workers', args.num_workers)
        args.checkpoint_dir = config['training'].get('checkpoint_dir', args.checkpoint_dir)
    
    return args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Training manager for Hybrid SNN-ANN detector."""
    
    def __init__(self, args):
        self.args = args
        
        # Set seed
        set_seed(args.seed)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Create model
        self.model = build_model(
            num_classes=args.num_classes,
            num_time_bins=args.num_time_bins,
            tau_init=args.tau_init,
            asab_kernel_size=args.asab_kernel_size
        ).to(self.device)
        
        total_params = self.model.get_param_count()
        print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Expected (paper): ~6.6M")
        
        # Create datasets
        print(f"\nLoading datasets from: {args.data_root}")
        
        self.train_dataset = Gen1Dataset(
            root_dir=args.data_root,
            split='train',
            num_time_bins=args.num_time_bins,
            height=args.height,
            width=args.width,
            augment=True
        )
        
        self.val_dataset = Gen1Dataset(
            root_dir=args.data_root,
            split='test',  # Gen1 uses 'test' for validation
            num_time_bins=args.num_time_bins,
            height=args.height,
            width=args.width,
            augment=False
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Loss function
        self.loss_fn = YOLOXLoss(
            num_classes=args.num_classes,
            strides=[8, 16, 32]
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler (OneCycle as per paper)
        steps_per_epoch = len(self.train_loader)
        if steps_per_epoch > 0:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=args.learning_rate,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1e4
            )
        else:
            self.scheduler = None
        
        # Mixed precision training
        self.use_amp = not args.no_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Evaluator
        self.evaluator = MAPEvaluator(num_classes=args.num_classes)
        
        # Checkpoint directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = Path(args.checkpoint_dir) / timestamp
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.checkpoint_dir / 'logs')
        
        # Save config
        self._save_config()
        
        # Resume from checkpoint
        self.start_epoch = 0
        self.best_map = 0.0
        self.global_step = 0
        
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _save_config(self):
        """Save training configuration."""
        config = vars(self.args)
        config_path = self.checkpoint_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Config saved to: {config_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_map = checkpoint.get('best_map', 0.0)
        self.global_step = checkpoint.get('global_step', 0)
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_map': self.best_map,
            'global_step': self.global_step
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            print(f"  → New best model saved! mAP: {self.best_map:.4f}")
        
        # Save periodic
        if (epoch + 1) % self.args.save_freq == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch:03d}.pth')
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_reg_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            events = batch['events'].to(self.device)  # [T, B, 2, H, W]
            labels = [l.to(self.device) for l in batch['labels']]
            image_sizes = batch['image_sizes']
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(events)
                    losses = self.loss_fn(outputs, labels, image_sizes)
                    loss = losses['total']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(events)
                losses = self.loss_fn(outputs, labels, image_sizes)
                loss = losses['total']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_reg_loss += losses['reg'].item()
            total_obj_loss += losses['obj'].item()
            total_cls_loss += losses['cls'].item()
            
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.2e}'
            })
            
            # Log to TensorBoard
            if batch_idx % self.args.log_freq == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/RegLoss', losses['reg'].item(), self.global_step)
                self.writer.add_scalar('Train/ObjLoss', losses['obj'].item(), self.global_step)
                self.writer.add_scalar('Train/ClsLoss', losses['cls'].item(), self.global_step)
                self.writer.add_scalar('Train/LR', lr, self.global_step)
            
            self.global_step += 1
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'reg': total_reg_loss / n_batches,
            'obj': total_obj_loss / n_batches,
            'cls': total_cls_loss / n_batches
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate model."""
        self.model.eval()
        self.evaluator.reset()
        
        pbar = tqdm(self.val_loader, desc='Validating')
        
        for batch in pbar:
            events = batch['events'].to(self.device)
            labels = batch['labels']
            
            # Get predictions
            if self.use_amp:
                with autocast():
                    predictions = self.model.predict(events, score_thresh=0.01)
            else:
                predictions = self.model.predict(events, score_thresh=0.01)
            
            # Update evaluator
            self.evaluator.update(predictions, labels)
        
        # Compute metrics
        metrics = self.evaluator.compute()
        
        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(f'Val/{name}', value, epoch)
        
        return metrics
    
    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training for {self.args.epochs} epochs")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(reg: {train_metrics['reg']:.4f}, "
                  f"obj: {train_metrics['obj']:.4f}, "
                  f"cls: {train_metrics['cls']:.4f})")
            
            # Validate
            if (epoch + 1) % self.args.eval_freq == 0 or epoch == self.args.epochs - 1:
                val_metrics = self.validate(epoch)
                
                print(f"  Val mAP@0.5: {val_metrics['mAP_50']:.4f}")
                print(f"  Val mAP: {val_metrics['mAP']:.4f}")
                
                # Check for best model
                is_best = val_metrics['mAP'] > self.best_map
                if is_best:
                    self.best_map = val_metrics['mAP']
            else:
                is_best = False
            
            # Save checkpoint
            self._save_checkpoint(epoch, is_best)
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best mAP: {self.best_map:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*60}")
        
        self.writer.close()


def main():
    args = parse_args()
    
    # Load config if provided
    if args.config is not None:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
        args = merge_config_with_args(args, config)
    
    # Print configuration
    print("\nConfiguration:")
    print("-" * 40)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 40)
    
    # Create trainer and start training
    trainer = Trainer(args)
    
    if len(trainer.train_loader) == 0:
        print("\nNo training data found!")
        print("Please download the Gen1 dataset and update the data_root path.")
        print("Dataset URL: https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/")
        return
    
    trainer.train()


if __name__ == '__main__':
    main()
