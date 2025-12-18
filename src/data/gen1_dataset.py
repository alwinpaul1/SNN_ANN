"""
Gen1 Automotive Detection Dataset Loader

Dataset specifications (from paper):
- Resolution: 304 × 240 (H × W)
- Recording duration: 39 hours
- Classes: 2 (car, pedestrian) - NOT 3!
- Format: Events stored in .dat or .h5 files
- Annotations: Bounding boxes with timestamps

Training setup (from paper Section 4.1):
- Event representations from 5ms bins
- Detection every 50ms using last 10 time bins
- Data augmentation: random horizontal flips, zoom, crop

Paper Reference: Ahmed et al. (arXiv:2403.10173v4)
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .event_representation import EventRepresentation


# Gen1 class mapping (CORRECTED: only 2 classes!)
GEN1_CLASSES = {
    'car': 0,
    'pedestrian': 1
}
GEN1_NUM_CLASSES = 2


class Gen1Dataset(Dataset):
    """
    Gen1 Automotive Detection Dataset
    
    Dataset structure (typical):
    gen1/
    ├── train/
    │   ├── 17-04-06_15-19-21_cut_000_239000000_295000000/
    │   │   ├── events.h5 or events_td.dat
    │   │   └── labels_v2.npy
    │   └── ...
    └── test/
        └── ...
    
    Each sequence contains:
    - events: (x, y, t, p) tuples
    - labels: bounding boxes with (t, x, y, w, h, class_id)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        num_time_bins: int = 10,
        bin_duration_ms: float = 5.0,
        detection_interval_ms: float = 50.0,
        height: int = 304,
        width: int = 240,
        augment: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            root_dir: Path to Gen1 dataset root
            split: 'train', 'val', or 'test'
            num_time_bins: Number of temporal bins (T=10)
            bin_duration_ms: Duration per bin (5ms)
            detection_interval_ms: Detection interval (50ms)
            height: Frame height (304)
            width: Frame width (240)
            augment: Enable data augmentation
            max_samples: Limit number of samples (for debugging)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_time_bins = num_time_bins
        self.bin_duration_ms = bin_duration_ms
        self.detection_interval_ms = detection_interval_ms
        self.height = height
        self.width = width
        self.augment = augment and (split == 'train')
        
        # Event representation converter
        self.event_repr = EventRepresentation(
            height=height,
            width=width,
            num_time_bins=num_time_bins,
            bin_duration_ms=bin_duration_ms
        )
        
        # Load dataset index
        self.samples = self._load_samples()
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from Gen1 {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """
        Load dataset samples.
        
        Each sample contains:
        - sequence_path: Path to sequence directory
        - t_start: Start timestamp
        - t_end: End timestamp
        - labels: List of bounding boxes at t_end
        """
        samples = []
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            print(f"Warning: Split directory not found: {split_dir}")
            print("Please download the Gen1 dataset and extract it to the correct location.")
            return samples
        
        # Iterate through sequences
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            # Find events file
            events_path = None
            for events_name in ['events.h5', 'events_td.dat', 'events.npy', 'events.dat']:
                candidate = seq_dir / events_name
                if candidate.exists():
                    events_path = candidate
                    break
            
            if events_path is None:
                continue
            
            # Find labels file
            labels_path = None
            for labels_name in ['labels_v2.npy', 'labels.npy', 'bbox.npy', 'labels.h5']:
                candidate = seq_dir / labels_name
                if candidate.exists():
                    labels_path = candidate
                    break
            
            # Also check for _bbox.npy files
            if labels_path is None:
                for f in seq_dir.iterdir():
                    if f.name.endswith('_bbox.npy'):
                        labels_path = f
                        break
            
            if labels_path is None:
                continue
            
            # Load labels to get detection timestamps
            try:
                labels = self._load_labels(labels_path)
            except Exception as e:
                print(f"Warning: Could not load labels from {labels_path}: {e}")
                continue
            
            if len(labels) == 0:
                continue
            
            # Get unique timestamps with labels
            unique_times = np.unique(labels[:, 0])
            
            # Create samples for each detection window
            window_us = self.detection_interval_ms * 1000  # Convert to microseconds
            
            for t_label in unique_times:
                t_end = t_label
                t_start = t_end - window_us
                
                if t_start < 0:
                    continue
                
                # Get labels at this timestamp
                mask = labels[:, 0] == t_label
                frame_labels = labels[mask, 1:]  # Remove timestamp column
                
                samples.append({
                    'sequence_dir': str(seq_dir),
                    'events_path': str(events_path),
                    't_start': t_start,
                    't_end': t_end,
                    'labels': frame_labels  # [N, 5]: x, y, w, h, class_id
                })
        
        return samples
    
    def _load_labels(self, labels_path: Path) -> np.ndarray:
        """
        Load bounding box labels.
        
        Returns:
            labels: [N, 6] array with (t, x, y, w, h, class_id)
        """
        labels_path = str(labels_path)
        
        if labels_path.endswith('.npy'):
            labels = np.load(labels_path, allow_pickle=True)
            
            # Handle structured array
            if labels.dtype.names is not None:
                # Convert structured array to regular array
                columns = []
                for name in ['t', 'x', 'y', 'w', 'h', 'class_id']:
                    if name in labels.dtype.names:
                        columns.append(labels[name])
                    elif name == 'class_id' and 'class_confidence' in labels.dtype.names:
                        # Some datasets use class_confidence instead
                        columns.append(labels['class_confidence'])
                
                if len(columns) >= 5:
                    if len(columns) == 5:
                        # Add class_id as 0 if not present
                        columns.append(np.zeros(len(labels), dtype=np.float32))
                    labels = np.column_stack(columns)
                else:
                    raise ValueError(f"Could not parse label columns: {labels.dtype.names}")
            
            # Handle different label formats
            if labels.ndim == 1:
                # Try to convert object array
                labels = np.array([list(l) if hasattr(l, '__iter__') else l for l in labels])
        
        elif labels_path.endswith('.h5'):
            with h5py.File(labels_path, 'r') as f:
                labels = f['labels'][:]
        else:
            raise ValueError(f"Unknown labels format: {labels_path}")
        
        return labels.astype(np.float32)
    
    def _load_events(
        self,
        events_path: str,
        t_start: float,
        t_end: float
    ) -> np.ndarray:
        """
        Load events from file within time window.
        
        Returns:
            events: [N, 4] array with (x, y, t, p)
        """
        if events_path.endswith('.h5'):
            with h5py.File(events_path, 'r') as f:
                # Try different key formats
                if 'events' in f:
                    events_group = f['events']
                    if 'x' in events_group:
                        x = events_group['x'][:]
                        y = events_group['y'][:]
                        t = events_group['t'][:]
                        p = events_group['p'][:]
                    else:
                        # Events stored as single array
                        data = events_group[:]
                        x, y, t, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
                else:
                    # Try root level keys
                    x = f['x'][:] if 'x' in f else f['events/x'][:]
                    y = f['y'][:] if 'y' in f else f['events/y'][:]
                    t = f['t'][:] if 't' in f else f['events/t'][:]
                    p = f['p'][:] if 'p' in f else f['events/p'][:]
        
        elif events_path.endswith('.npy'):
            data = np.load(events_path)
            if data.dtype.names is not None:
                x = data['x']
                y = data['y']
                t = data['t']
                p = data['p']
            else:
                x, y, t, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        
        elif events_path.endswith('.dat'):
            # Load prophesee dat format
            events = self._load_dat_events(events_path)
            if len(events) == 0:
                return np.zeros((0, 4), dtype=np.float32)
            x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        
        else:
            raise ValueError(f"Unknown events format: {events_path}")
        
        # Filter by time window
        mask = (t >= t_start) & (t < t_end)
        events = np.column_stack([x[mask], y[mask], t[mask], p[mask]])
        
        return events.astype(np.float32)
    
    def _load_dat_events(self, dat_path: str) -> np.ndarray:
        """Load events from Prophesee .dat format."""
        try:
            # Try metavision SDK first
            from metavision_core.event_io import EventsIterator
            events_iterator = EventsIterator(dat_path)
            events_list = []
            for ev in events_iterator:
                events_list.append(ev)
            
            if not events_list:
                return np.zeros((0, 4), dtype=np.float32)
            
            events = np.concatenate(events_list)
            return np.column_stack([
                events['x'], events['y'], events['t'], events['p']
            ])
        
        except ImportError:
            # Fallback: try to read as raw binary
            try:
                return self._parse_dat_file(dat_path)
            except Exception as e:
                print(f"Warning: Could not load .dat file {dat_path}: {e}")
                print("Install metavision SDK for full .dat support: pip install metavision-sdk-base")
                return np.zeros((0, 4), dtype=np.float32)
    
    def _parse_dat_file(self, dat_path: str) -> np.ndarray:
        """
        Basic parser for Prophesee .dat files.
        
        Note: This is a simplified implementation. For full support,
        use the metavision SDK.
        """
        with open(dat_path, 'rb') as f:
            # Read header
            header = b''
            while True:
                line = f.readline()
                if line.startswith(b'%'):
                    header += line
                else:
                    break
            
            # Parse events
            # Format depends on the specific .dat version
            # This is a basic implementation
            dtype = np.dtype([
                ('t', '<u4'),
                ('x', '<u2'),
                ('y', '<u2'),
                ('p', '<u1'),
            ])
            
            try:
                data = np.fromfile(f, dtype=dtype)
                if len(data) > 0:
                    return np.column_stack([
                        data['x'], data['y'], data['t'], data['p']
                    ]).astype(np.float32)
            except:
                pass
        
        return np.zeros((0, 4), dtype=np.float32)
    
    def _apply_augmentation(
        self,
        event_tensor: torch.Tensor,
        labels: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Apply data augmentation.
        
        Paper: "random horizontal flips, zoom, and crop"
        """
        if not self.augment:
            return event_tensor, labels
        
        labels = labels.copy()
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            event_tensor = torch.flip(event_tensor, dims=[-1])  # Flip width
            if len(labels) > 0:
                # Flip x coordinates: x_new = width - x - w
                labels[:, 0] = self.width - labels[:, 0] - labels[:, 2]
        
        # Random zoom (scale 0.8-1.2)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            
            T, C, H, W = event_tensor.shape
            new_H, new_W = int(H * scale), int(W * scale)
            
            if new_H > 0 and new_W > 0:
                # Reshape for interpolation: [T*C, 1, H, W]
                event_flat = event_tensor.reshape(T * C, 1, H, W)
                event_scaled = torch.nn.functional.interpolate(
                    event_flat, size=(new_H, new_W), mode='nearest'
                )
                
                # Center crop or pad back to original size
                if scale > 1:
                    # Center crop
                    start_h = (new_H - H) // 2
                    start_w = (new_W - W) // 2
                    event_scaled = event_scaled[:, :, start_h:start_h+H, start_w:start_w+W]
                else:
                    # Zero pad
                    pad_h = (H - new_H) // 2
                    pad_w = (W - new_W) // 2
                    event_padded = torch.zeros(T * C, 1, H, W, dtype=event_tensor.dtype)
                    event_padded[:, :, pad_h:pad_h+new_H, pad_w:pad_w+new_W] = event_scaled
                    event_scaled = event_padded
                
                event_tensor = event_scaled.reshape(T, C, H, W)
                
                # Scale labels
                if len(labels) > 0:
                    labels[:, :4] *= scale
        
        return event_tensor, labels
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            dict with:
            - 'events': [T, 2, H, W] event tensor
            - 'labels': [N, 5] bounding boxes (x, y, w, h, class_id)
            - 'image_size': (H, W)
        """
        sample = self.samples[idx]
        
        # Load events
        events = self._load_events(
            sample['events_path'],
            sample['t_start'],
            sample['t_end']
        )
        
        # Convert to tensor
        event_tensor = self.event_repr(
            events,
            t_start_us=sample['t_start'],
            t_end_us=sample['t_end']
        )
        
        # Get labels
        labels = sample['labels'].copy()  # [N, 5]: x, y, w, h, class_id
        
        # Apply augmentation
        event_tensor, labels = self._apply_augmentation(event_tensor, labels)
        
        # Convert labels to tensor
        if len(labels) > 0:
            labels = torch.from_numpy(labels).float()
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        
        return {
            'events': event_tensor,  # [T, 2, H, W]
            'labels': labels,        # [N, 5]
            'image_size': (self.height, self.width)
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length labels.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Collated batch with:
        - 'events': [T, B, 2, H, W]
        - 'labels': List of [N_i, 5] tensors
        - 'image_sizes': List of (H, W) tuples
    """
    events = torch.stack([item['events'] for item in batch])  # [B, T, 2, H, W]
    
    # Permute to [T, B, 2, H, W] for model
    events = events.permute(1, 0, 2, 3, 4)
    
    labels = [item['labels'] for item in batch]  # List of [N_i, 5] tensors
    image_sizes = [item['image_size'] for item in batch]
    
    return {
        'events': events,
        'labels': labels,
        'image_sizes': image_sizes
    }


# Testing code
if __name__ == '__main__':
    print("Testing Gen1 Dataset Implementation...")
    print("="*60)
    
    # Test with synthetic data if dataset not available
    print("\nCreating synthetic test data...")
    
    import tempfile
    import shutil
    
    # Create temporary dataset structure
    tmp_dir = tempfile.mkdtemp()
    train_dir = Path(tmp_dir) / 'train' / 'seq_001'
    train_dir.mkdir(parents=True)
    
    # Generate synthetic events
    n_events = 5000
    events = np.column_stack([
        np.random.randint(0, 240, n_events),  # x
        np.random.randint(0, 304, n_events),  # y
        np.sort(np.random.randint(0, 100000, n_events)),  # t
        np.random.randint(0, 2, n_events)     # p
    ]).astype(np.float32)
    np.save(train_dir / 'events.npy', events)
    
    # Generate synthetic labels
    n_labels = 10
    labels = np.column_stack([
        np.array([50000, 50000, 50000, 50000, 50000,
                  100000, 100000, 100000, 100000, 100000]),  # t
        np.random.randint(10, 200, n_labels),  # x
        np.random.randint(10, 280, n_labels),  # y
        np.random.randint(20, 50, n_labels),   # w
        np.random.randint(30, 60, n_labels),   # h
        np.random.randint(0, 2, n_labels)      # class_id
    ]).astype(np.float32)
    np.save(train_dir / 'labels_v2.npy', labels)
    
    try:
        # Test dataset
        dataset = Gen1Dataset(
            root_dir=tmp_dir,
            split='train',
            num_time_bins=10,
            augment=False
        )
        
        print(f"\nDataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            # Get a sample
            sample = dataset[0]
            
            print(f"\nSample contents:")
            print(f"  Events shape: {sample['events'].shape}")
            print(f"  Labels shape: {sample['labels'].shape}")
            print(f"  Image size: {sample['image_size']}")
            
            # Test collate function
            batch = collate_fn([dataset[i] for i in range(min(2, len(dataset)))])
            
            print(f"\nBatch contents:")
            print(f"  Events shape: {batch['events'].shape}")
            print(f"  Labels: {[l.shape for l in batch['labels']]}")
            
            # Verify shapes
            assert batch['events'].shape[0] == 10, "T dimension should be 10"
            assert batch['events'].shape[2] == 2, "Polarity channels should be 2"
            
            print("\n✓ Gen1 Dataset tests passed!")
        else:
            print("\nNo samples found - this is expected for synthetic test")
    
    finally:
        # Cleanup
        shutil.rmtree(tmp_dir)
