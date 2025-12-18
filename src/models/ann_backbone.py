"""
ANN Backbone - Corrected per Paper Table 9

Architecture (from paper):
| Layer | Kernel     | Output Dimensions          | Type |
|-------|------------|----------------------------|------|
| 6     | 256c3p1s1  | 256 × H/8 × W/8           | ANN  |
| 7     | 256c3p1s2  | 256 × H/16 × W/16         | ANN  |
| 9     | 256c3p1s1  | 256 × H/16 × W/16         | ANN  |
| 10    | 256c3p1s2  | 256 × H/32 × W/32         | ANN  |

Note: Layers 8 and 11 are DWConvLSTM (only in +RNN variant)
For base hybrid model, we skip to layers 9-10.

CORRECTION: Previous implementation had all stride=1!
- Correct strides: s1, s2, s1, s2

Paper Reference: Ahmed et al. (arXiv:2403.10173v4)
"""

import torch
import torch.nn as nn
from typing import Dict


class ANNConvBlock(nn.Module):
    """
    Standard ANN Convolutional Block: Conv2D → BatchNorm2D → ReLU
    
    Paper Section 3.2:
    "Each ANN block consists of standard convolution operations, 
    normalization, and ReLU activation functions"
    """
    
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
        """
        Args:
            x: [B, C, H, W]
        Returns:
            output: [B, C_out, H_out, W_out]
        """
        return self.relu(self.bn(self.conv(x)))


class ANNBackbone(nn.Module):
    """
    ANN Backbone for high-level spatial feature extraction.
    
    CORRECTED Architecture per Paper Table 9:
    - Layer 6: 256→256, stride=1
    - Layer 7: 256→256, stride=2
    - Layer 9: 256→256, stride=1  
    - Layer 10: 256→256, stride=2
    
    Outputs multi-scale features for FPN detection head:
    - P3: 1/8 scale (H/8 × W/8)
    - P4: 1/16 scale (H/16 × W/16)
    - P5: 1/32 scale (H/32 × W/32)
    """
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        
        # Layer 6: 256c3p1s1 → 256 × H/8 × W/8
        self.block1 = ANNConvBlock(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Layer 7: 256c3p1s2 → 256 × H/16 × W/16
        self.block2 = ANNConvBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=2,  # CORRECTED: stride 2!
            padding=1
        )
        
        # Layer 9: 256c3p1s1 → 256 × H/16 × W/16
        self.block3 = ANNConvBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Layer 10: 256c3p1s2 → 256 × H/32 × W/32
        self.block4 = ANNConvBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=2,  # CORRECTED: stride 2!
            padding=1
        )
        
        # Output channels for each scale
        self.out_channels = {
            'p3': 256,
            'p4': 256,
            'p5': 256
        }
    
    def forward(self, F_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            F_out: Dense features from ASAB [B, 256, H/8, W/8]
                  For Gen1: [B, 256, 38, 30]
        
        Returns:
            features: Dict of multi-scale features for FPN
                - 'p3': [B, 256, H/8, W/8]   (38×30 for Gen1)
                - 'p4': [B, 256, H/16, W/16] (19×15 for Gen1)
                - 'p5': [B, 256, H/32, W/32] (9×7 for Gen1)
        """
        # Layer 6: [B, 256, 38, 30] → [B, 256, 38, 30]
        feat1 = self.block1(F_out)
        
        # Layer 7: [B, 256, 38, 30] → [B, 256, 19, 15]
        feat2 = self.block2(feat1)
        
        # Layer 9: [B, 256, 19, 15] → [B, 256, 19, 15]
        feat3 = self.block3(feat2)
        
        # Layer 10: [B, 256, 19, 15] → [B, 256, 9, 7]
        feat4 = self.block4(feat3)
        
        return {
            'p3': feat1,  # 1/8 scale
            'p4': feat3,  # 1/16 scale
            'p5': feat4,  # 1/32 scale
        }
    
    def get_param_count(self) -> int:
        """Returns parameter count for this component."""
        return sum(p.numel() for p in self.parameters())


# Testing code
if __name__ == '__main__':
    print("Testing ANN Backbone Implementation...")
    
    # Create model
    ann = ANNBackbone(in_channels=256)
    
    # Input: [B, C, H, W] - output from ASAB (H/8 × W/8)
    # For Gen1 (304×240): H/8=38, W/8=30
    B, C, H, W = 2, 256, 38, 30
    x = torch.randn(B, C, H, W)
    
    print(f"Input shape: {x.shape}")
    print(f"Expected outputs:")
    print(f"  - p3: [B, 256, {H}, {W}] = [{B}, 256, {H}, {W}]")
    print(f"  - p4: [B, 256, {H//2}, {W//2}] = [{B}, 256, {H//2}, {W//2}]")
    print(f"  - p5: [B, 256, {H//4}, {W//4}] = [{B}, 256, {H//4}, {W//4}]")
    
    # Forward pass
    with torch.no_grad():
        features = ann(x)
    
    print(f"\nActual outputs:")
    for name, feat in features.items():
        print(f"  - {name}: {feat.shape}")
    
    # Verify dimensions
    assert features['p3'].shape == (B, 256, H, W), "p3 shape mismatch"
    assert features['p4'].shape == (B, 256, H // 2, W // 2), "p4 shape mismatch"
    assert features['p5'].shape == (B, 256, H // 4, W // 4), "p5 shape mismatch"
    
    # Parameter count
    param_count = ann.get_param_count()
    print(f"\nANN Backbone Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"Expected: ~1.8M (per paper)")
    
    print("\n✓ ANN Backbone tests passed!")
