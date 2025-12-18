"""
Attention-based SNN-ANN Bridge (ASAB) Module

Components:
1. Channel-wise Temporal Grouping
2. Time-wise Separable Deformable Convolution (TSDC) - Φ_tsdc
3. Temporal Attention - Φ_ta  
4. Event-Rate Spatial (ERS) Attention

Key equations from paper:
- Eq (3): Ak = ωk(Asc), Aq = Ψ(ωq(Asc)), Av = Ψ(ωv(Asc))
- Eq (4): Ascore = softmax(Aq @ Ak) ∈ R^(T×T)
- Eq (5): Aattended = Ψ(Av @ Ascore)
- Eq (6): E_feature = sigmoid(S_rate) ⊙ A_out

Paper Reference: Ahmed et al. (arXiv:2403.10173v4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 for capturing irregular spatial structures.
    
    The paper states this helps capture "irregular spatial spike-structure"
    better than standard square grid kernels.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,  # Paper: kernel_size=5 for TSDC
        padding: int = 2,
        groups: int = 1
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main convolution weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
        # Offset prediction network (predicts 2*k*k offsets per location)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=1)
        )
        
        # Initialize offsets to zero (start with regular grid)
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            output: [B, C_out, H, W]
        """
        # Predict offsets from input features
        offsets = self.offset_conv(x)  # [B, 2*k*k, H, W]
        
        # Apply deformable convolution
        output = deform_conv2d(
            input=x,
            offset=offsets,
            weight=self.weight,
            bias=self.bias,
            padding=self.padding,
        )
        
        return output


class TimeWiseSeparableDeformableConv(nn.Module):
    """
    Time-wise Separable Deformable Convolution (TSDC) - Φ_tsdc
    
    From paper Section 3.3.1:
    "We implemented the TSDC as a time-wise separable convolution to capture 
    spatial details independently of time-based changes"
    
    CRITICAL CORRECTION:
    Paper says "we set the group of deformable convolution kernels equal to 
    the number of time steps T" - NOT channels!
    """
    
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Deformable conv applied per time step
        self.deform_conv = DeformableConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1
        )
    
    def forward(self, A_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_in: [B, C, T, H, W] - After channel-wise temporal grouping
        
        Returns:
            A_sc: Spatial context [B, C, T, H, W]
        """
        B, C, T, H, W = A_in.shape
        
        spatial_contexts = []
        
        # Process each time step independently (time-wise separable)
        for t in range(T):
            # Extract time slice: [B, C, H, W]
            x_t = A_in[:, :, t, :, :]
            
            # Apply deformable conv to capture local spatial context
            sc_t = self.deform_conv(x_t)  # [B, C, H, W]
            
            spatial_contexts.append(sc_t)
        
        # Stack back: [B, C, T, H, W]
        A_sc = torch.stack(spatial_contexts, dim=2)
        
        return A_sc


class TemporalAttention(nn.Module):
    """
    Temporal Attention Module - Φ_ta
    
    From paper Section 3.3.1, Equations (3-5):
    
    1. Compute Q, K, V using 1×1 convolutions:
       Ak = ωk(Asc) ∈ R^(H'W' × T)
       Aq = Ψ(ωq(Asc)) ∈ R^(T × H'W')
       Av = Ψ(ωv(Asc)) ∈ R^(T × H'W')
    
    2. Compute attention scores:
       Ascore = softmax(Aq @ Ak) ∈ R^(T × T)
    
    3. Apply attention:
       Aattended = Ψ(Av @ Ascore) ∈ R^(T × H' × W')
    
    4. Final output via 1×1 conv weighted sum across time
    """
    
    def __init__(self, channels: int, num_time_bins: int = 10):
        super().__init__()
        
        self.channels = channels
        self.num_time_bins = num_time_bins
        
        # 1×1 convolutions for Q, K, V projection
        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Final 1×1 conv for weighted sum across time
        self.out_conv = nn.Conv2d(channels * num_time_bins, channels, kernel_size=1)
    
    def forward(self, A_sc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_sc: Spatial context [B, C, T, H, W]
        
        Returns:
            A_out: Attended features [B, C, H, W]
        """
        B, C, T, H, W = A_sc.shape
        
        # Reshape to apply 1×1 conv: [B*T, C, H, W]
        A_sc_flat = A_sc.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # Compute Q, K, V
        Q = self.conv_q(A_sc_flat)  # [B*T, C, H, W]
        K = self.conv_k(A_sc_flat)  # [B*T, C, H, W]
        V = self.conv_v(A_sc_flat)  # [B*T, C, H, W]
        
        # Reshape for attention computation
        # Q: [B, T, C, H*W] → [B, C, T, H*W] → [B*C, T, H*W]
        Q = Q.reshape(B, T, C, H * W).permute(0, 2, 1, 3).reshape(B * C, T, H * W)
        
        # K: [B, T, C, H*W] → [B, C, H*W, T] → [B*C, H*W, T]
        K = K.reshape(B, T, C, H * W).permute(0, 2, 3, 1).reshape(B * C, H * W, T)
        
        # V: [B, T, C, H*W] → [B, C, T, H*W] → [B*C, T, H*W]
        V = V.reshape(B, T, C, H * W).permute(0, 2, 1, 3).reshape(B * C, T, H * W)
        
        # Eq (4): Ascore = softmax(Aq @ Ak) ∈ R^(T×T)
        # Q: [B*C, T, H*W], K: [B*C, H*W, T] → Ascore: [B*C, T, T]
        A_score = torch.bmm(Q, K)  # [B*C, T, T]
        A_score = F.softmax(A_score / (H * W) ** 0.5, dim=-1)
        
        # Eq (5): Aattended = V @ Ascore^T
        # V: [B*C, T, H*W], Ascore: [B*C, T, T] → [B*C, T, H*W]
        A_attended = torch.bmm(A_score, V)  # [B*C, T, H*W]
        
        # Reshape: [B*C, T, H*W] → [B, C, T, H, W]
        A_attended = A_attended.reshape(B, C, T, H, W)
        
        # Final weighted sum across time using 1×1 conv
        # Reshape to [B, C*T, H, W]
        A_attended_flat = A_attended.reshape(B, C * T, H, W)
        
        A_out = self.out_conv(A_attended_flat)  # [B, C, H, W]
        
        return A_out


class SpatialAwareTemporalAttention(nn.Module):
    """
    Spatial-Aware Temporal (SAT) Attention
    
    Combines:
    1. Channel-wise Temporal Grouping (reshape operation)
    2. Time-wise Separable Deformable Convolution (TSDC)
    3. Temporal Attention
    """
    
    def __init__(self, channels: int, kernel_size: int = 5, num_time_bins: int = 10):
        super().__init__()
        
        self.tsdc = TimeWiseSeparableDeformableConv(channels, kernel_size)
        self.temporal_attention = TemporalAttention(channels, num_time_bins)
    
    def forward(self, E_spike: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E_spike: [T, B, C, H, W] (time-first from SNN)
        
        Returns:
            A_out: [B, C, H, W]
        """
        # Convert from [T, B, C, H, W] to [B, C, T, H, W]
        E_spike = E_spike.permute(1, 2, 0, 3, 4)  # → [B, C, T, H, W]
        
        # Step 1: Channel-wise temporal grouping (implicit - already in [B, C, T, H, W])
        A_in = E_spike
        
        # Step 2: Time-wise Separable Deformable Conv
        A_sc = self.tsdc(A_in)  # [B, C, T, H, W]
        
        # Step 3: Temporal Attention
        A_out = self.temporal_attention(A_sc)  # [B, C, H, W]
        
        return A_out


class EventRateSpatialAttention(nn.Module):
    """
    Event-Rate Spatial (ERS) Attention
    
    From paper Section 3.3.2, Equation (6):
    
    S_rate = Σ_t E_spike(t, :, :, :)   [sum over time]
    E_feature = sigmoid(S_rate) ⊙ A_out
    
    This weights spatial regions by their event activity.
    """
    
    def forward(self, E_spike: torch.Tensor, A_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E_spike: [T, B, C, H, W] (time-first format)
            A_out: SAT attention output [B, C, H, W]
        
        Returns:
            E_feature: [B, C, H, W]
        """
        # Sum over time (dim=0 since [T, B, C, H, W])
        S_rate = E_spike.sum(dim=0)  # [B, C, H, W]
        
        # Equation (6): E_feature = sigmoid(S_rate) ⊙ A_out
        spatial_weights = torch.sigmoid(S_rate)
        E_feature = spatial_weights * A_out
        
        return E_feature


class ASABModule(nn.Module):
    """
    Attention-based SNN-ANN Bridge (ASAB) Module - β_asab
    
    Converts sparse spike representations to dense features while
    preserving spatiotemporal information through:
    - SAT: Spatial-Aware Temporal Attention
    - ERS: Event-Rate Spatial Attention
    
    From paper:
    F_out = β_asab(E_spike) ∈ R^(C × H' × W')
    """
    
    def __init__(self, channels: int = 256, kernel_size: int = 5, num_time_bins: int = 10):
        """
        Args:
            channels: Number of channels (256 from SNN backbone)
            kernel_size: Kernel size for TSDC (paper uses 5)
            num_time_bins: Number of time bins T (paper uses 10)
        """
        super().__init__()
        
        self.sat_attention = SpatialAwareTemporalAttention(channels, kernel_size, num_time_bins)
        self.ers_attention = EventRateSpatialAttention()
    
    def forward(self, E_spike: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E_spike: Spike tensor from SNN [T, B, C, H, W]
        
        Returns:
            F_out: Dense feature map [B, C, H, W]
        """
        # SAT: Spatial-Aware Temporal Attention
        A_out = self.sat_attention(E_spike)  # [B, C, H, W]
        
        # ERS: Event-Rate Spatial Attention
        F_out = self.ers_attention(E_spike, A_out)  # [B, C, H, W]
        
        return F_out
    
    def get_param_count(self) -> int:
        """Returns parameter count for this component."""
        return sum(p.numel() for p in self.parameters())


# Testing code
if __name__ == '__main__':
    print("Testing ASAB Module Implementation...")
    
    # Create module
    asab = ASABModule(channels=256, kernel_size=5, num_time_bins=10)
    
    # Input: [T, B, C, H, W] - output from SNN backbone
    T, B, C, H, W = 10, 2, 256, 38, 30  # Gen1 dimensions after SNN
    E_spike = torch.randn(T, B, C, H, W)
    
    print(f"Input shape: {E_spike.shape}")
    print(f"Expected output: [B, C, H, W] = [{B}, {C}, {H}, {W}]")
    
    # Forward pass
    with torch.no_grad():
        F_out = asab(E_spike)
    
    print(f"Output shape: {F_out.shape}")
    
    # Verify dimensions
    assert F_out.shape == (B, C, H, W), \
        f"Shape mismatch! Expected {(B, C, H, W)}, got {F_out.shape}"
    
    # Parameter count
    param_count = asab.get_param_count()
    print(f"\nASAB Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"Expected: ~0.8M (per paper)")
    
    # Test gradient flow
    E_spike_grad = torch.randn(T, B, C, H, W, requires_grad=True)
    F_out_grad = asab(E_spike_grad)
    loss = F_out_grad.sum()
    loss.backward()
    print(f"\nGradient check - Input grad exists: {E_spike_grad.grad is not None}")
    
    print("\n✓ ASAB Module tests passed!")
