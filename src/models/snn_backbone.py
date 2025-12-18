"""
SNN Backbone - Corrected per Paper Table 9

Architecture (from paper):
| Layer | Kernel     | Output Dimensions      | Type |
|-------|------------|------------------------|------|
| 1     | 64c3p1s2   | T × 64 × H/2 × W/2    | SNN  |
| 2     | 128c3p1s2  | T × 128 × H/4 × W/4   | SNN  |
| 3     | 256c3p1s2  | T × 256 × H/8 × W/8   | SNN  |
| 4     | 256c3p1s1  | T × 256 × H/8 × W/8   | SNN  |

CORRECTION: Previous implementation used wrong channel counts and strides!
- Channels: 64→128→256→256 (NOT 32→64→128→256)
- Strides: s2,s2,s2,s1 (NOT s1,s2,s1,s2)

Paper Reference: Ahmed et al. (arXiv:2403.10173v4)
"""

import torch
import torch.nn as nn
from .plif_neuron import PLIFNeuronMultiStep


class SNNConvBlock(nn.Module):
    """
    SNN Convolutional Block: Conv2D → BatchNorm2D → PLIF Neuron
    
    Processes spatiotemporal data with spiking activations.
    The Conv and BN are applied per time step, PLIF integrates across time.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        tau_init: float = 2.0
    ):
        super().__init__()
        
        # Spatial convolution (applied per time step)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # No bias when using BatchNorm
        )
        
        # Batch normalization (applied per time step)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # PLIF spiking neuron (integrates across time)
        self.neuron = PLIFNeuronMultiStep(tau_init=tau_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [T, B, C, H, W] (time-first format)
        
        Returns:
            spikes: Output spikes [T, B, C_out, H_out, W_out]
        """
        T, B, C, H, W = x.shape
        
        # Reshape for batch processing: [T*B, C, H, W]
        x_flat = x.reshape(T * B, C, H, W)
        
        # Apply conv + bn
        x_conv = self.conv(x_flat)
        x_bn = self.bn(x_conv)
        
        # Reshape back: [T, B, C_out, H_out, W_out]
        _, C_out, H_out, W_out = x_bn.shape
        x_temporal = x_bn.reshape(T, B, C_out, H_out, W_out)
        
        # Apply PLIF neuron across time
        spikes = self.neuron(x_temporal)
        
        return spikes


class SNNBackbone(nn.Module):
    """
    SNN Backbone for low-level spatiotemporal feature extraction.
    
    CORRECTED Architecture per Paper Table 9:
    - 4 SNN blocks with specific channel counts and strides
    - Channels: 64 → 128 → 256 → 256
    - Strides: 2, 2, 2, 1
    - Output: T × 256 × H/8 × W/8
    
    For Gen1 (304×240): Output is T × 256 × 38 × 30
    """
    
    def __init__(self, in_channels: int = 2, tau_init: float = 2.0):
        """
        Args:
            in_channels: Input channels (2 for event polarity)
            tau_init: Initial tau for PLIF neurons
        """
        super().__init__()
        
        # Layer 1: 2 → 64, stride=2
        # Output: T × 64 × H/2 × W/2
        self.block1 = SNNConvBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=2,  # CORRECTED: stride 2
            padding=1,
            tau_init=tau_init
        )
        
        # Layer 2: 64 → 128, stride=2
        # Output: T × 128 × H/4 × W/4
        self.block2 = SNNConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,  # stride 2
            padding=1,
            tau_init=tau_init
        )
        
        # Layer 3: 128 → 256, stride=2
        # Output: T × 256 × H/8 × W/8
        self.block3 = SNNConvBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,  # stride 2
            padding=1,
            tau_init=tau_init
        )
        
        # Layer 4: 256 → 256, stride=1
        # Output: T × 256 × H/8 × W/8
        self.block4 = SNNConvBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,  # CORRECTED: stride 1 (not 2!)
            padding=1,
            tau_init=tau_init
        )
        
        # Output channels
        self.out_channels = 256
    
    def forward(self, event_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            event_tensor: [T, B, 2, H, W] - Event tensor with polarity channels
                         For Gen1: [T, B, 2, 304, 240]
        
        Returns:
            E_spike: Spike tensor [T, B, 256, H/8, W/8]
                    For Gen1: [T, B, 256, 38, 30]
        """
        # Block 1: [T, B, 2, 304, 240] → [T, B, 64, 152, 120]
        x = self.block1(event_tensor)
        
        # Block 2: [T, B, 64, 152, 120] → [T, B, 128, 76, 60]
        x = self.block2(x)
        
        # Block 3: [T, B, 128, 76, 60] → [T, B, 256, 38, 30]
        x = self.block3(x)
        
        # Block 4: [T, B, 256, 38, 30] → [T, B, 256, 38, 30]
        E_spike = self.block4(x)
        
        return E_spike
    
    def get_param_count(self) -> int:
        """Returns parameter count for this component."""
        return sum(p.numel() for p in self.parameters())


# Testing code
if __name__ == '__main__':
    print("Testing SNN Backbone Implementation...")
    
    # Create model
    snn = SNNBackbone(in_channels=2, tau_init=2.0)
    
    # Gen1 input dimensions: T=10, B=2, C=2, H=304, W=240
    T, B, C, H, W = 10, 2, 2, 304, 240
    x = torch.randn(T, B, C, H, W)
    
    print(f"Input shape: {x.shape}")
    print(f"Expected output: [T, B, 256, H/8, W/8] = [{T}, {B}, 256, {H//8}, {W//8}]")
    
    # Forward pass
    with torch.no_grad():
        out = snn(x)
    
    print(f"Output shape: {out.shape}")
    print(f"Spike rate: {out.mean().item():.4f}")
    
    # Verify dimensions
    expected_h, expected_w = H // 8, W // 8
    assert out.shape == (T, B, 256, expected_h, expected_w), \
        f"Shape mismatch! Expected {(T, B, 256, expected_h, expected_w)}, got {out.shape}"
    
    # Parameter count
    param_count = snn.get_param_count()
    print(f"\nSNN Backbone Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"Expected: ~1.2M (per paper)")
    
    print("\n✓ SNN Backbone tests passed!")
