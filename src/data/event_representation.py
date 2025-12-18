"""
Event Representation - Following Paper Section 3.1

Creates 4D tensor Events[tk-1, tk] ∈ R^(T×2×H×W)
- T: Number of time discretization steps (10 bins of 5ms each)
- 2: Polarity channels (positive and negative events)
- H×W: Spatial dimensions (304×240 for Gen1)

Paper notes:
- "Event representations for the SNN are constructed from 5 ms bins"
- "During training, object detections are generated every 50 ms, 
   using the SNN's output from the last 10 time bins"

Paper Equation (1):
Events[t_k−1, t_k](t, p, x, y) = Σ δ(p − pn)δ(x − xn)δ(y − yn)δ(t − t'n)
where t'n = floor((tn - ta)/(tb - ta) · T)

Paper Reference: Ahmed et al. (arXiv:2403.10173v4)
"""

import numpy as np
import torch
from typing import Tuple, Optional

# Try to import numba for acceleration
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available. Event conversion will be slower.")


if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def _events_to_tensor_numba(
        x: np.ndarray,
        y: np.ndarray,
        t_bins: np.ndarray,
        p: np.ndarray,
        tensor: np.ndarray,
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Numba-accelerated event to tensor conversion.
        Counts events per (time_bin, polarity, y, x) cell.
        """
        n_events = len(x)
        
        for i in prange(n_events):
            xi = int(x[i])
            yi = int(y[i])
            ti = int(t_bins[i])
            pi = int(p[i])
            
            if 0 <= xi < width and 0 <= yi < height:
                tensor[ti, pi, yi, xi] += 1
        
        return tensor


def _events_to_tensor_numpy(
    x: np.ndarray,
    y: np.ndarray,
    t_bins: np.ndarray,
    p: np.ndarray,
    tensor: np.ndarray,
    height: int,
    width: int
) -> np.ndarray:
    """
    NumPy implementation of event to tensor conversion.
    Slower than numba but works without compilation.
    """
    # Filter valid events
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid].astype(np.int32)
    y = y[valid].astype(np.int32)
    t_bins = t_bins[valid].astype(np.int32)
    p = p[valid].astype(np.int32)
    
    # Use np.add.at for accumulation
    indices = (t_bins, p, y, x)
    np.add.at(tensor, indices, 1)
    
    return tensor


class EventRepresentation:
    """
    Converts raw events to tensor representation.
    
    Paper uses:
    - T = 10 time bins
    - Bin duration = 5ms (so 50ms total window)
    - 2 polarity channels
    - Gen1 resolution: H=304, W=240
    """
    
    def __init__(
        self,
        height: int = 304,
        width: int = 240,
        num_time_bins: int = 10,
        bin_duration_ms: float = 5.0,
        normalize: bool = True
    ):
        """
        Args:
            height: Event frame height (304 for Gen1)
            width: Event frame width (240 for Gen1)
            num_time_bins: Number of temporal bins (T=10 in paper)
            bin_duration_ms: Duration of each bin in ms (5ms in paper)
            normalize: Whether to normalize event counts
        """
        self.height = height
        self.width = width
        self.num_time_bins = num_time_bins
        self.bin_duration_ms = bin_duration_ms
        self.total_duration_ms = num_time_bins * bin_duration_ms  # 50ms
        self.normalize = normalize
    
    def __call__(
        self,
        events: np.ndarray,
        t_start_us: Optional[float] = None,
        t_end_us: Optional[float] = None
    ) -> torch.Tensor:
        """
        Convert events to tensor representation.
        
        Args:
            events: Array of events [N, 4] with columns (x, y, t, p)
                   - x, y: pixel coordinates
                   - t: timestamp in microseconds
                   - p: polarity (0 or 1)
            t_start_us: Start time in microseconds (optional)
            t_end_us: End time in microseconds (optional)
        
        Returns:
            event_tensor: [T, 2, H, W] tensor
        """
        if len(events) == 0:
            return torch.zeros(
                (self.num_time_bins, 2, self.height, self.width),
                dtype=torch.float32
            )
        
        # Extract event components
        x = events[:, 0].astype(np.float32)
        y = events[:, 1].astype(np.float32)
        t = events[:, 2].astype(np.float64)  # timestamps in microseconds
        p = events[:, 3].astype(np.int32)
        
        # Ensure polarity is 0 or 1
        p = np.clip(p, 0, 1)
        
        # Determine time window
        if t_start_us is None:
            t_start_us = t.min()
        if t_end_us is None:
            t_end_us = t.max()
        
        # Filter events within time window
        mask = (t >= t_start_us) & (t < t_end_us)
        x, y, t, p = x[mask], y[mask], t[mask], p[mask]
        
        if len(x) == 0:
            return torch.zeros(
                (self.num_time_bins, 2, self.height, self.width),
                dtype=torch.float32
            )
        
        # Normalize time to [0, T) and discretize
        # Paper Eq (1): t'_n = floor((t_n - t_a) / (t_b - t_a) * T)
        t_normalized = (t - t_start_us) / (t_end_us - t_start_us + 1e-6)
        t_bins = np.clip(
            (t_normalized * self.num_time_bins).astype(np.int32),
            0, self.num_time_bins - 1
        )
        
        # Create tensor
        tensor = np.zeros(
            (self.num_time_bins, 2, self.height, self.width),
            dtype=np.float32
        )
        
        # Use numba if available, otherwise numpy
        if NUMBA_AVAILABLE:
            tensor = _events_to_tensor_numba(
                x, y, t_bins, p, tensor,
                self.height, self.width
            )
        else:
            tensor = _events_to_tensor_numpy(
                x, y, t_bins, p, tensor,
                self.height, self.width
            )
        
        # Normalize if requested
        if self.normalize:
            # Per-channel normalization
            for t_idx in range(self.num_time_bins):
                for p_idx in range(2):
                    channel = tensor[t_idx, p_idx]
                    if channel.max() > 0:
                        tensor[t_idx, p_idx] = channel / (channel.max() + 1e-6)
        
        return torch.from_numpy(tensor)
    
    def events_to_frame(
        self,
        events: np.ndarray,
        accumulate: bool = True
    ) -> np.ndarray:
        """
        Convert events to single accumulated frame (for visualization).
        
        Args:
            events: [N, 4] array (x, y, t, p)
            accumulate: If True, accumulate counts; if False, binary
        
        Returns:
            frame: [H, W, 3] RGB image (Blue: negative, Red: positive)
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if len(events) == 0:
            return frame
        
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        p = events[:, 3].astype(np.int32)
        
        # Valid coordinates
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x, y, p = x[valid], y[valid], p[valid]
        
        if accumulate:
            # Accumulate event counts
            pos_count = np.zeros((self.height, self.width), dtype=np.float32)
            neg_count = np.zeros((self.height, self.width), dtype=np.float32)
            
            pos_mask = p == 1
            neg_mask = p == 0
            
            np.add.at(pos_count, (y[pos_mask], x[pos_mask]), 1)
            np.add.at(neg_count, (y[neg_mask], x[neg_mask]), 1)
            
            # Normalize to 0-255
            if pos_count.max() > 0:
                frame[:, :, 2] = (pos_count / pos_count.max() * 255).astype(np.uint8)  # Red
            if neg_count.max() > 0:
                frame[:, :, 0] = (neg_count / neg_count.max() * 255).astype(np.uint8)  # Blue
        else:
            # Binary visualization
            pos_mask = p == 1
            neg_mask = p == 0
            
            frame[y[pos_mask], x[pos_mask], 2] = 255  # Red
            frame[y[neg_mask], x[neg_mask], 0] = 255  # Blue
        
        return frame
    
    def events_to_voxel_grid(
        self,
        events: np.ndarray,
        t_start_us: Optional[float] = None,
        t_end_us: Optional[float] = None
    ) -> torch.Tensor:
        """
        Alternative voxel grid representation with linear interpolation.
        
        This distributes each event's contribution to neighboring time bins
        using bilinear interpolation in time.
        
        Args:
            events: [N, 4] array (x, y, t, p)
            t_start_us: Start time in microseconds
            t_end_us: End time in microseconds
        
        Returns:
            voxel: [T, 2, H, W] tensor with interpolated counts
        """
        if len(events) == 0:
            return torch.zeros(
                (self.num_time_bins, 2, self.height, self.width),
                dtype=torch.float32
            )
        
        x = events[:, 0].astype(np.float32)
        y = events[:, 1].astype(np.float32)
        t = events[:, 2].astype(np.float64)
        p = events[:, 3].astype(np.int32)
        
        if t_start_us is None:
            t_start_us = t.min()
        if t_end_us is None:
            t_end_us = t.max()
        
        # Normalize time to [0, T-1]
        t_normalized = (t - t_start_us) / (t_end_us - t_start_us + 1e-6) * (self.num_time_bins - 1)
        
        # Get integer and fractional parts
        t_floor = np.floor(t_normalized).astype(np.int32)
        t_frac = t_normalized - t_floor
        
        # Clip to valid range
        t_floor = np.clip(t_floor, 0, self.num_time_bins - 2)
        
        # Create voxel grid
        voxel = np.zeros(
            (self.num_time_bins, 2, self.height, self.width),
            dtype=np.float32
        )
        
        # Valid spatial coordinates
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        x = x[valid].astype(np.int32)
        y = y[valid].astype(np.int32)
        t_floor = t_floor[valid]
        t_frac = t_frac[valid]
        p = np.clip(p[valid], 0, 1)
        
        # Distribute to neighboring bins
        for i in range(len(x)):
            xi, yi, ti, pi = x[i], y[i], t_floor[i], p[i]
            frac = t_frac[i]
            
            voxel[ti, pi, yi, xi] += (1 - frac)
            if ti + 1 < self.num_time_bins:
                voxel[ti + 1, pi, yi, xi] += frac
        
        return torch.from_numpy(voxel)


# Testing code
if __name__ == '__main__':
    print("Testing Event Representation Implementation...")
    
    # Create representation converter
    event_repr = EventRepresentation(
        height=304,
        width=240,
        num_time_bins=10,
        bin_duration_ms=5.0
    )
    
    # Generate synthetic events
    n_events = 10000
    np.random.seed(42)
    
    events = np.column_stack([
        np.random.randint(0, 240, n_events),  # x
        np.random.randint(0, 304, n_events),  # y
        np.sort(np.random.randint(0, 50000, n_events)),  # t (microseconds, sorted)
        np.random.randint(0, 2, n_events)     # p (polarity)
    ]).astype(np.float32)
    
    print(f"\nSynthetic events shape: {events.shape}")
    print(f"Time range: {events[:, 2].min():.0f} - {events[:, 2].max():.0f} µs")
    
    # Convert to tensor
    tensor = event_repr(events)
    
    print(f"\nOutput tensor shape: {tensor.shape}")
    print(f"Expected: [T, 2, H, W] = [10, 2, 304, 240]")
    
    # Verify dimensions
    assert tensor.shape == (10, 2, 304, 240), f"Shape mismatch: {tensor.shape}"
    
    # Check statistics
    print(f"\nTensor statistics:")
    print(f"  Min: {tensor.min().item():.4f}")
    print(f"  Max: {tensor.max().item():.4f}")
    print(f"  Mean: {tensor.mean().item():.4f}")
    print(f"  Non-zero: {(tensor > 0).sum().item()}")
    
    # Test voxel grid with interpolation
    voxel = event_repr.events_to_voxel_grid(events)
    print(f"\nVoxel grid shape: {voxel.shape}")
    print(f"  Non-zero: {(voxel > 0).sum().item()}")
    
    # Test visualization
    frame = event_repr.events_to_frame(events)
    print(f"\nVisualization frame shape: {frame.shape}")
    print(f"  Red channel (positive events): {frame[:,:,2].sum()}")
    print(f"  Blue channel (negative events): {frame[:,:,0].sum()}")
    
    print("\n✓ Event Representation tests passed!")
