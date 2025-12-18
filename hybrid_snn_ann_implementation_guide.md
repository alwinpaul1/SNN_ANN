# Comprehensive Implementation Guide: Hybrid SNN-ANN for Event-Based Object Detection
## Based on Ahmed et al. (arXiv:2403.10173v4) - Gen1 Dataset Only

---

## 🔴 Critical Corrections from Original Implementation

Before diving into the implementation, here are the **critical errors** found in the previous implementation that have been corrected:

| Component | Previous (Wrong) | Corrected (Paper) |
|-----------|-----------------|-------------------|
| **PLIF Equation** | `v_mem = v_mem * (1-decay) + x` | `V[t] = V[t-1] + (1/τ)(X[t] - (V[t-1] - Vreset))` |
| **τ computation** | `τ = sigmoid(w)` | `τ = 1/sigmoid(w)` (inverse!) |
| **SNN channels** | 32→64→128→256 | **64→128→256→256** (Table 9) |
| **SNN strides** | s1,s2,s1,s2 | **s2,s2,s2,s1** |
| **ANN strides** | All s1 | **s1,s2,s1,s2** |
| **TSDC groups** | `groups=in_channels` | **`groups=T`** (time steps) |
| **Temporal Attention** | Channel-wise complex | **Simple T×T softmax attention** |
| **Gen1 classes** | 3 classes | **2 classes** (car, pedestrian) |
| **Output resolution** | H/4×W/4 | **H/8×W/8 → H/32×W/32** (multi-scale) |

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup with UV](#2-environment-setup-with-uv)
3. [Core Components](#3-core-components)
   - 3.1 [PLIF Neuron (Corrected)](#31-plif-neuron-corrected)
   - 3.2 [SNN Backbone (Corrected Architecture)](#32-snn-backbone-corrected-architecture)
   - 3.3 [ASAB Bridge Module](#33-asab-bridge-module)
   - 3.4 [ANN Backbone (Corrected Architecture)](#34-ann-backbone-corrected-architecture)
   - 3.5 [Complete Hybrid Detector](#35-complete-hybrid-detector)
4. [Event Representation](#4-event-representation)
5. [Gen1 Dataset Loader](#5-gen1-dataset-loader)
6. [Detection Head (YOLOX)](#6-detection-head-yolox)
7. [Training Pipeline](#7-training-pipeline)
8. [Configuration](#8-configuration)
9. [Expected Results](#9-expected-results)

---

## 1. Project Structure

```
hybrid_snn_ann/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   └── gen1_config.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── plif_neuron.py          # PLIF neuron with surrogate gradients
│   │   ├── snn_blocks.py           # SNN convolutional blocks
│   │   ├── snn_backbone.py         # 4-layer SNN backbone
│   │   ├── asab_module.py          # Attention-based SNN-ANN Bridge
│   │   ├── ann_backbone.py         # 4-layer ANN backbone
│   │   ├── hybrid_backbone.py      # Combined SNN-ASAB-ANN
│   │   ├── detection_head.py       # YOLOX detection head
│   │   └── hybrid_detector.py      # Complete model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── event_representation.py # Event to tensor conversion
│   │   ├── gen1_dataset.py         # Gen1 dataset loader
│   │   └── augmentations.py        # Data augmentation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py              # mAP computation
│   │   ├── losses.py               # YOLOX losses
│   │   └── visualization.py        # Detection visualization
│   └── train.py                    # Training script
├── scripts/
│   ├── download_gen1.sh            # Dataset download
│   └── evaluate.py                 # Evaluation script
└── checkpoints/
    └── .gitkeep
```

---

## 2. Environment Setup with UV

### 2.1 Install UV (if not already installed)

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 2.2 Project Initialization

```bash
# Create project directory
mkdir hybrid_snn_ann && cd hybrid_snn_ann

# Initialize UV project
uv init

# Create virtual environment with Python 3.10
uv venv --python 3.10
```

### 2.3 pyproject.toml

```toml
[project]
name = "hybrid-snn-ann"
version = "0.1.0"
description = "Hybrid SNN-ANN for Event-Based Object Detection (Ahmed et al.)"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "numpy>=1.24.0",
    "spikingjelly>=0.0.0.0.14",
    "h5py>=3.9.0",
    "opencv-python>=4.8.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0.1",
    "tensorboard>=2.15.0",
    "scipy>=1.11.0",
    "pycocotools>=2.0.7",
    "einops>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
```

### 2.4 Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install all dependencies with UV
uv pip install -e .

# Install PyTorch with CUDA (for A100)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install spikingjelly (specific version for compatibility)
uv pip install spikingjelly==0.0.0.0.14

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import spikingjelly; print(f'SpikingJelly: {spikingjelly.__version__}')"
```

---

## 3. Core Components

### 3.1 PLIF Neuron (Corrected)

**File: `src/models/plif_neuron.py`**

The paper defines PLIF dynamics in **Equation (2)**:

$$V[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset}))$$

where $\tau = \text{sigmoid}(w)^{-1}$ (the **inverse** of sigmoid).

```python
"""
PLIF Neuron Implementation - Corrected per Paper Equation (2)

Key Correction: τ = 1/sigmoid(w), NOT sigmoid(w)
The original implementation had this backwards!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateGradient(torch.autograd.Function):
    """
    Surrogate gradient function for spike generation.
    Uses the rectangular function as surrogate (following SpikingJelly).
    
    Forward: Heaviside step function (threshold at 1.0)
    Backward: Rectangular surrogate gradient
    """
    
    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        # Spike if membrane potential exceeds threshold
        return (membrane_potential >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        membrane_potential, = ctx.saved_tensors
        threshold = ctx.threshold
        
        # Rectangular surrogate gradient
        # Width of the surrogate: 1.0 (can be tuned)
        grad = (torch.abs(membrane_potential - threshold) < 0.5).float()
        
        return grad * grad_output, None


class PLIFNeuron(nn.Module):
    """
    Parametric Leaky Integrate-and-Fire (PLIF) Neuron
    
    Paper Equation (2):
        V[t] = V[t-1] + (1/τ) * (X[t] - (V[t-1] - V_reset))
    
    where:
        τ = sigmoid(w)^(-1)  # CRITICAL: It's the INVERSE!
        w is a learnable parameter
    
    Args:
        tau_init: Initial value for the learnable time constant parameter w
        v_threshold: Spike threshold (default: 1.0)
        v_reset: Reset potential after spike (default: 0.0)
        detach_reset: Whether to detach reset in backward pass
    """
    
    def __init__(
        self,
        tau_init: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True
    ):
        super().__init__()
        
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        
        # Learnable time constant parameter
        # Initialize such that sigmoid(w) gives reasonable τ
        # If tau_init=2.0, we want sigmoid(w)^(-1) ≈ 2.0, so sigmoid(w) ≈ 0.5, w ≈ 0
        init_w = torch.log(torch.tensor(1.0 / tau_init) / (1 - 1.0 / tau_init))
        self.w = nn.Parameter(torch.tensor(init_w))
        
        # Surrogate gradient function
        self.spike_fn = SurrogateGradient.apply
    
    @property
    def tau(self) -> torch.Tensor:
        """
        Compute τ = sigmoid(w)^(-1) = 1/sigmoid(w)
        
        CRITICAL CORRECTION: Paper says τ = sigmoid(w)^(-1)
        This is the INVERSE of sigmoid, not sigmoid itself!
        """
        return 1.0 / torch.sigmoid(self.w)
    
    def forward(self, x: torch.Tensor, v_prev: torch.Tensor = None) -> tuple:
        """
        Single time step forward pass.
        
        Args:
            x: Input current [B, C, H, W]
            v_prev: Previous membrane potential [B, C, H, W] or None
        
        Returns:
            spike: Output spikes [B, C, H, W]
            v_new: New membrane potential [B, C, H, W]
        """
        if v_prev is None:
            v_prev = torch.zeros_like(x)
        
        # Paper Equation (2): V[t] = V[t-1] + (1/τ)(X[t] - (V[t-1] - V_reset))
        tau = self.tau
        v_new = v_prev + (1.0 / tau) * (x - (v_prev - self.v_reset))
        
        # Generate spike using surrogate gradient
        spike = self.spike_fn(v_new, self.v_threshold)
        
        # Reset: hard reset after spike
        if self.detach_reset:
            v_new = v_new * (1 - spike.detach()) + self.v_reset * spike.detach()
        else:
            v_new = v_new * (1 - spike) + self.v_reset * spike
        
        return spike, v_new


class PLIFNeuronMultiStep(nn.Module):
    """
    Multi-step PLIF neuron that processes entire temporal sequence.
    
    This is more efficient for training as it processes all time steps at once.
    """
    
    def __init__(
        self,
        tau_init: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True
    ):
        super().__init__()
        self.neuron = PLIFNeuron(tau_init, v_threshold, v_reset, detach_reset)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process entire temporal sequence.
        
        Args:
            x: Input tensor [T, B, C, H, W] (time-first format)
        
        Returns:
            spikes: Output spikes [T, B, C, H, W]
        """
        T = x.shape[0]
        spikes = []
        v = None
        
        for t in range(T):
            spike, v = self.neuron(x[t], v)
            spikes.append(spike)
        
        return torch.stack(spikes, dim=0)  # [T, B, C, H, W]
```

### 3.2 SNN Backbone (Corrected Architecture)

**File: `src/models/snn_backbone.py`**

**CRITICAL: Paper Table 9 specifies exact architecture:**
- Layer 1: 64 channels, stride 2 → Output: T×64×H/2×W/2
- Layer 2: 128 channels, stride 2 → Output: T×128×H/4×W/4  
- Layer 3: 256 channels, stride 2 → Output: T×256×H/8×W/8
- Layer 4: 256 channels, stride 1 → Output: T×256×H/8×W/8

```python
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
```

### 3.3 ASAB Bridge Module

**File: `src/models/asab_module.py`**

The ASAB module contains two attention mechanisms:
1. **SAT (Spatial-Aware Temporal) Attention**: Captures temporal relationships using spatial context
2. **ERS (Event-Rate Spatial) Attention**: Weights spatial regions by event activity

```python
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
        
        # Main convolution weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
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
        
        # Deformable conv with groups=T (set dynamically in forward)
        # We create a single deformable conv and apply it per time step
        self.deform_conv = DeformableConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1  # Apply to all channels, but separately per time
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
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        
        # 1×1 convolutions for Q, K, V projection
        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Final 1×1 conv for weighted sum across time
        # This will be created dynamically based on T
        self.out_conv = None
    
    def forward(self, A_sc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_sc: Spatial context [B, C, T, H, W]
        
        Returns:
            A_out: Attended features [B, C, H, W]
        """
        B, C, T, H, W = A_sc.shape
        
        # Process each channel group independently
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
        
        # Create output conv if needed (handles varying T)
        if self.out_conv is None or self.out_conv.in_channels != C * T:
            self.out_conv = nn.Conv2d(C * T, C, kernel_size=1).to(A_attended.device)
        
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
    
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        
        self.tsdc = TimeWiseSeparableDeformableConv(channels, kernel_size)
        self.temporal_attention = TemporalAttention(channels)
    
    def forward(self, E_spike: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E_spike: [T, B, C, H, W] or [B, C, T, H, W]
        
        Returns:
            A_out: [B, C, H, W]
        """
        # Ensure format is [B, C, T, H, W]
        if E_spike.dim() == 5:
            if E_spike.shape[0] < E_spike.shape[1]:  # T < B unlikely, assume [T, B, C, H, W]
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
            E_spike: [T, B, C, H, W] or [B, C, T, H, W]
            A_out: SAT attention output [B, C, H, W]
        
        Returns:
            E_feature: [B, C, H, W]
        """
        # Ensure format for summing over time
        if E_spike.dim() == 5:
            if E_spike.shape[0] < E_spike.shape[1]:  # [T, B, C, H, W]
                # Sum over time (dim=0)
                S_rate = E_spike.sum(dim=0)  # [B, C, H, W]
            else:  # [B, C, T, H, W]
                S_rate = E_spike.sum(dim=2)  # [B, C, H, W]
        else:
            raise ValueError(f"Expected 5D tensor, got {E_spike.dim()}D")
        
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
    
    def __init__(self, channels: int = 256, kernel_size: int = 5):
        """
        Args:
            channels: Number of channels (256 from SNN backbone)
            kernel_size: Kernel size for TSDC (paper uses 5)
        """
        super().__init__()
        
        self.sat_attention = SpatialAwareTemporalAttention(channels, kernel_size)
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
```

### 3.4 ANN Backbone (Corrected Architecture)

**File: `src/models/ann_backbone.py`**

**CORRECTED per Paper Table 9:**
- Layer 6: 256c3p1s1 → 256×H/8×W/8
- Layer 7: 256c3p1s2 → 256×H/16×W/16 (stride 2!)
- Layer 9: 256c3p1s1 → 256×H/16×W/16
- Layer 10: 256c3p1s2 → 256×H/32×W/32 (stride 2!)

```python
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
"""

import torch
import torch.nn as nn


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
    
    Outputs multi-scale features for FPN detection head.
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
    
    def forward(self, F_out: torch.Tensor) -> dict:
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
```

### 3.5 Complete Hybrid Detector

**File: `src/models/hybrid_detector.py`**

```python
"""
Complete Hybrid SNN-ANN Detector

Architecture Flow (from paper Figure 1):
Events → To Tensor → SNN Blocks → β_asab → ANN Blocks → Detection FPN + Head → Detections

This implements the base Hybrid model (without DWConvLSTM).
For +RNN variant, add DWConvLSTM between ANN blocks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .snn_backbone import SNNBackbone
from .asab_module import ASABModule
from .ann_backbone import ANNBackbone


class HybridBackbone(nn.Module):
    """
    Hybrid SNN-ANN Backbone
    
    Combines:
    - SNN: Low-level spatiotemporal feature extraction
    - ASAB: Attention-based bridge (sparse→dense conversion)
    - ANN: High-level spatial feature extraction
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        tau_init: float = 2.0,
        asab_kernel_size: int = 5
    ):
        super().__init__()
        
        # SNN Backbone: extracts low-level spatiotemporal features
        self.snn = SNNBackbone(in_channels=in_channels, tau_init=tau_init)
        
        # ASAB Bridge: converts sparse spikes to dense features
        self.asab = ASABModule(channels=256, kernel_size=asab_kernel_size)
        
        # ANN Backbone: extracts high-level spatial features
        self.ann = ANNBackbone(in_channels=256)
    
    def forward(self, event_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            event_tensor: [T, B, 2, H, W] - Event representation
        
        Returns:
            features: Multi-scale feature dict for detection head
        """
        # SNN: Extract spatiotemporal features
        # [T, B, 2, H, W] → [T, B, 256, H/8, W/8]
        E_spike = self.snn(event_tensor)
        
        # ASAB: Bridge sparse→dense
        # [T, B, 256, H/8, W/8] → [B, 256, H/8, W/8]
        F_out = self.asab(E_spike)
        
        # ANN: Extract high-level features
        # [B, 256, H/8, W/8] → multi-scale features
        features = self.ann(F_out)
        
        return features


class YOLOXHead(nn.Module):
    """
    YOLOX Detection Head (simplified)
    
    Paper uses YOLOX framework for detection.
    This is a simplified implementation - for full YOLOX, use the official repo.
    """
    
    def __init__(self, num_classes: int = 2, in_channels: int = 256):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Shared stem for all scales
        self.stems = nn.ModuleDict()
        self.cls_convs = nn.ModuleDict()
        self.reg_convs = nn.ModuleDict()
        self.cls_preds = nn.ModuleDict()
        self.reg_preds = nn.ModuleDict()
        self.obj_preds = nn.ModuleDict()
        
        for scale in ['p3', 'p4', 'p5']:
            # Stem conv
            self.stems[scale] = nn.Sequential(
                nn.Conv2d(in_channels, 256, 1, 1, 0),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            
            # Classification branch
            self.cls_convs[scale] = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            
            # Regression branch
            self.reg_convs[scale] = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            
            # Prediction layers
            self.cls_preds[scale] = nn.Conv2d(256, num_classes, 1)
            self.reg_preds[scale] = nn.Conv2d(256, 4, 1)  # x, y, w, h
            self.obj_preds[scale] = nn.Conv2d(256, 1, 1)  # objectness
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Multi-scale feature dict {'p3', 'p4', 'p5'}
        
        Returns:
            outputs: Detection outputs per scale
        """
        outputs = {}
        
        for scale in ['p3', 'p4', 'p5']:
            x = features[scale]
            
            # Stem
            x = self.stems[scale](x)
            
            # Classification branch
            cls_feat = self.cls_convs[scale](x)
            cls_pred = self.cls_preds[scale](cls_feat)
            
            # Regression branch
            reg_feat = self.reg_convs[scale](x)
            reg_pred = self.reg_preds[scale](reg_feat)
            obj_pred = self.obj_preds[scale](reg_feat)
            
            outputs[scale] = {
                'cls': cls_pred,  # [B, num_classes, H, W]
                'reg': reg_pred,  # [B, 4, H, W]
                'obj': obj_pred   # [B, 1, H, W]
            }
        
        return outputs


class HybridSNNANNDetector(nn.Module):
    """
    Complete Hybrid SNN-ANN Object Detector
    
    For Gen1 dataset:
    - Input: Event tensor [T, B, 2, 304, 240]
    - Output: Detections (bboxes, classes, scores)
    - Classes: 2 (car, pedestrian)
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # CORRECTED: Gen1 has 2 classes
        in_channels: int = 2,
        tau_init: float = 2.0,
        asab_kernel_size: int = 5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Hybrid backbone
        self.backbone = HybridBackbone(
            in_channels=in_channels,
            tau_init=tau_init,
            asab_kernel_size=asab_kernel_size
        )
        
        # Detection head
        self.head = YOLOXHead(num_classes=num_classes, in_channels=256)
    
    def forward(
        self,
        event_tensor: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Args:
            event_tensor: [T, B, 2, H, W]
        
        Returns:
            outputs: Detection outputs per scale
        """
        # Extract features
        features = self.backbone(event_tensor)
        
        # Detection head
        outputs = self.head(features)
        
        return outputs
    
    def get_param_count(self) -> int:
        """Returns total parameter count (paper reports 6.6M)"""
        return sum(p.numel() for p in self.parameters())
```

---

## 4. Event Representation

**File: `src/data/event_representation.py`**

From paper Section 3.1, Equation (1):

$$\text{Events}[t_k{-}1, t_k](t, p, x, y) = \sum_{e_n \in E} \delta(p - p_n)\delta(x - x_n)\delta(y - y_n)\delta(t - t'_n)$$

where $t'_n = \lfloor \frac{t_n - t_a}{t_b - t_a} \cdot T \rfloor$

```python
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
"""

import numpy as np
import torch
from typing import Tuple, Optional
import numba
from numba import jit


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
    
    for i in numba.prange(n_events):
        xi = int(x[i])
        yi = int(y[i])
        ti = int(t_bins[i])
        pi = int(p[i])
        
        if 0 <= xi < width and 0 <= yi < height:
            tensor[ti, pi, yi, xi] += 1
    
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
        bin_duration_ms: float = 5.0
    ):
        """
        Args:
            height: Event frame height (304 for Gen1)
            width: Event frame width (240 for Gen1)
            num_time_bins: Number of temporal bins (T=10 in paper)
            bin_duration_ms: Duration of each bin in ms (5ms in paper)
        """
        self.height = height
        self.width = width
        self.num_time_bins = num_time_bins
        self.bin_duration_ms = bin_duration_ms
        self.total_duration_ms = num_time_bins * bin_duration_ms  # 50ms
    
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
        
        # Create tensor using numba-accelerated function
        tensor = np.zeros(
            (self.num_time_bins, 2, self.height, self.width),
            dtype=np.float32
        )
        
        tensor = _events_to_tensor_numba(
            x, y, t_bins, p, tensor,
            self.height, self.width
        )
        
        return torch.from_numpy(tensor)
    
    def events_to_frame(self, events: np.ndarray) -> np.ndarray:
        """
        Convert events to single accumulated frame (for visualization).
        
        Args:
            events: [N, 4] array
        
        Returns:
            frame: [H, W, 3] RGB image
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
        
        # Positive events: red, Negative events: blue
        pos_mask = p == 1
        neg_mask = p == 0
        
        frame[y[pos_mask], x[pos_mask], 2] = 255  # Red
        frame[y[neg_mask], x[neg_mask], 0] = 255  # Blue
        
        return frame
```

---

## 5. Gen1 Dataset Loader

**File: `src/data/gen1_dataset.py`**

```python
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
            split: 'train' or 'test'
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
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Iterate through sequences
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            # Find events file
            events_path = None
            for events_name in ['events.h5', 'events_td.dat', 'events.npy']:
                candidate = seq_dir / events_name
                if candidate.exists():
                    events_path = candidate
                    break
            
            if events_path is None:
                continue
            
            # Find labels file
            labels_path = None
            for labels_name in ['labels_v2.npy', 'labels.npy', 'labels.h5']:
                candidate = seq_dir / labels_name
                if candidate.exists():
                    labels_path = candidate
                    break
            
            if labels_path is None:
                continue
            
            # Load labels to get detection timestamps
            labels = self._load_labels(labels_path)
            
            if len(labels) == 0:
                continue
            
            # Get unique timestamps with labels
            unique_times = np.unique(labels[:, 0])
            
            # Create samples for each detection window
            window_us = self.detection_interval_ms * 1000  # Convert to microseconds
            
            for t_label in unique_times:
                t_end = t_label
                t_start = t_end - window_us
                
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
        if str(labels_path).endswith('.npy'):
            labels = np.load(labels_path, allow_pickle=True)
            # Handle structured array
            if labels.dtype.names is not None:
                # Convert structured array to regular array
                labels = np.column_stack([
                    labels['t'],
                    labels['x'],
                    labels['y'],
                    labels['w'],
                    labels['h'],
                    labels['class_id']
                ])
        elif str(labels_path).endswith('.h5'):
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
                x = f['events/x'][:]
                y = f['events/y'][:]
                t = f['events/t'][:]
                p = f['events/p'][:]
        elif events_path.endswith('.npy'):
            data = np.load(events_path)
            x, y, t, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        elif events_path.endswith('.dat'):
            # Load prophesee dat format
            events = self._load_dat_events(events_path)
            x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        else:
            raise ValueError(f"Unknown events format: {events_path}")
        
        # Filter by time window
        mask = (t >= t_start) & (t < t_end)
        events = np.column_stack([x[mask], y[mask], t[mask], p[mask]])
        
        return events.astype(np.float32)
    
    def _load_dat_events(self, dat_path: str) -> np.ndarray:
        """Load events from Prophesee .dat format."""
        # This is a simplified loader - use prophesee tools for full support
        try:
            from metavision_core.event_io import EventsIterator
            events_iterator = EventsIterator(dat_path)
            events = []
            for ev in events_iterator:
                events.append(ev)
            events = np.concatenate(events) if events else np.array([])
            return np.column_stack([
                events['x'], events['y'], events['t'], events['p']
            ])
        except ImportError:
            # Fallback: basic binary reading
            # Note: This is simplified and may not work for all .dat files
            raise NotImplementedError(
                "Please install metavision SDK for .dat file support: "
                "pip install metavision-sdk-base"
            )
    
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
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            event_tensor = torch.flip(event_tensor, dims=[-1])  # Flip width
            if len(labels) > 0:
                # Flip x coordinates: x_new = width - x - w
                labels = labels.copy()
                labels[:, 0] = self.width - labels[:, 0] - labels[:, 2]
        
        # Random zoom (scale 0.8-1.2)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            # Apply scaling via interpolation
            T, C, H, W = event_tensor.shape
            new_H, new_W = int(H * scale), int(W * scale)
            
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
                event_padded = torch.zeros(T * C, 1, H, W)
                event_padded[:, :, pad_h:pad_h+new_H, pad_w:pad_w+new_W] = event_scaled
                event_scaled = event_padded
            
            event_tensor = event_scaled.reshape(T, C, H, W)
            
            # Scale labels
            if len(labels) > 0:
                labels = labels.copy()
                labels[:, :4] *= scale  # Scale x, y, w, h
        
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
```

---

## 6. Detection Head (YOLOX)

**File: `src/utils/losses.py`**

```python
"""
YOLOX Loss Functions

Paper uses YOLOX framework with:
- IOU loss for bounding box regression
- Class loss (cross entropy)
- Objectness loss

Reference: YOLOX paper (Ge et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    x1y1x2y2: bool = False,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate IoU between boxes.
    
    Args:
        box1: [N, 4] boxes
        box2: [M, 4] boxes
        x1y1x2y2: If True, boxes are (x1, y1, x2, y2), else (cx, cy, w, h)
        eps: Small value to avoid division by zero
    
    Returns:
        iou: [N, M] IoU matrix
    """
    if not x1y1x2y2:
        # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Intersection
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area
    
    iou = inter_area / (union_area + eps)
    
    return iou


class YOLOXLoss(nn.Module):
    """
    YOLOX Loss combining:
    - IoU loss for regression
    - BCE loss for objectness
    - BCE loss for classification
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        strides: List[int] = [8, 16, 32],
        reg_weight: float = 5.0,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = strides
        self.reg_weight = reg_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
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
            targets: List of [N_i, 5] target boxes per image
            image_sizes: List of (H, W) tuples
        
        Returns:
            losses: Dict of loss values
        """
        device = next(iter(outputs.values()))['cls'].device
        
        total_loss = torch.tensor(0.0, device=device)
        reg_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        
        # Process each scale
        for scale_idx, (scale, scale_outputs) in enumerate(outputs.items()):
            stride = self.strides[scale_idx]
            
            cls_pred = scale_outputs['cls']  # [B, num_classes, H, W]
            reg_pred = scale_outputs['reg']  # [B, 4, H, W]
            obj_pred = scale_outputs['obj']  # [B, 1, H, W]
            
            B, _, H, W = cls_pred.shape
            
            # Create grid for anchor-free detection
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            grid = torch.stack([xv, yv], dim=-1).float()  # [H, W, 2]
            
            # Process each image in batch
            for b in range(B):
                target = targets[b]  # [N, 5]: x, y, w, h, class_id
                
                if len(target) == 0:
                    # No targets - only objectness loss
                    obj_loss += self.bce_loss(
                        obj_pred[b],
                        torch.zeros_like(obj_pred[b])
                    ).mean()
                    continue
                
                # Convert targets to grid coordinates
                target_xy = target[:, :2] / stride  # Center x, y
                target_wh = target[:, 2:4] / stride  # Width, height
                target_cls = target[:, 4].long()     # Class id
                
                # Find positive samples using center sampling
                # (simplified - full YOLOX uses SimOTA)
                for t_idx in range(len(target)):
                    tx, ty = target_xy[t_idx]
                    tw, th = target_wh[t_idx]
                    tc = target_cls[t_idx]
                    
                    # Get grid cell containing center
                    gx, gy = int(tx.clamp(0, W-1)), int(ty.clamp(0, H-1))
                    
                    # Regression target
                    pred_reg = reg_pred[b, :, gy, gx]  # [4]
                    target_reg = torch.tensor([tx - gx, ty - gy, tw, th], device=device)
                    
                    # IoU loss
                    iou = bbox_iou(
                        pred_reg.unsqueeze(0),
                        target_reg.unsqueeze(0)
                    )[0, 0]
                    reg_loss += (1 - iou)
                    
                    # Objectness target
                    obj_target = torch.zeros_like(obj_pred[b])
                    obj_target[0, gy, gx] = 1.0
                    obj_loss += self.bce_loss(obj_pred[b], obj_target).mean()
                    
                    # Classification target
                    cls_target = torch.zeros(self.num_classes, device=device)
                    cls_target[tc] = 1.0
                    cls_loss += self.bce_loss(
                        cls_pred[b, :, gy, gx],
                        cls_target
                    ).mean()
        
        # Combine losses
        total_loss = (
            self.reg_weight * reg_loss +
            self.obj_weight * obj_loss +
            self.cls_weight * cls_loss
        )
        
        return {
            'total': total_loss,
            'reg': reg_loss,
            'obj': obj_loss,
            'cls': cls_loss
        }
```

---

## 7. Training Pipeline

**File: `src/train.py`**

```python
"""
Training Script for Hybrid SNN-ANN Detector

Paper training setup (Section 4.1):
- Epochs: 50 for Gen1
- Batch size: 24 (paper used 4× 3090 GPUs)
- Learning rate: 2e-4
- Optimizer: Adam with OneCycle scheduler
- Training time: ~8 hours on 4× 3090 (expect ~4 hours on A100)
"""

import os
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

from models.hybrid_detector import HybridSNNANNDetector
from data.gen1_dataset import Gen1Dataset, collate_fn
from utils.losses import YOLOXLoss
from utils.metrics import compute_map


def parse_args():
    parser = argparse.ArgumentParser(description='Train Hybrid SNN-ANN Detector')
    parser.add_argument('--config', type=str, default='configs/gen1_config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()


class Trainer:
    """Training manager for Hybrid SNN-ANN detector."""
    
    def __init__(self, config_path: str, resume: str = None):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = HybridSNNANNDetector(
            num_classes=self.config['model']['num_classes'],
            in_channels=self.config['model']['in_channels'],
            tau_init=self.config['model']['tau_init'],
            asab_kernel_size=self.config['model']['asab_kernel_size']
        ).to(self.device)
        
        print(f"Model parameters: {self.model.get_param_count() / 1e6:.2f}M")
        
        # Create datasets
        self.train_dataset = Gen1Dataset(
            root_dir=self.config['data']['root_dir'],
            split='train',
            num_time_bins=self.config['data']['num_time_bins'],
            bin_duration_ms=self.config['data']['bin_duration_ms'],
            height=self.config['data']['height'],
            width=self.config['data']['width'],
            augment=True
        )
        
        self.val_dataset = Gen1Dataset(
            root_dir=self.config['data']['root_dir'],
            split='test',
            num_time_bins=self.config['data']['num_time_bins'],
            bin_duration_ms=self.config['data']['bin_duration_ms'],
            height=self.config['data']['height'],
            width=self.config['data']['width'],
            augment=False
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Loss function
        self.loss_fn = YOLOXLoss(
            num_classes=self.config['model']['num_classes'],
            strides=[8, 16, 32]
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler (OneCycle as per paper)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['training']['learning_rate'],
            epochs=self.config['training']['epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Checkpoint directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir']) / timestamp
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.checkpoint_dir / 'logs')
        
        # Resume from checkpoint
        self.start_epoch = 0
        self.best_map = 0.0
        
        if resume:
            self._load_checkpoint(resume)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_map = checkpoint.get('best_map', 0.0)
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
        
        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch}.pth')
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            events = batch['events'].to(self.device)  # [T, B, 2, H, W]
            labels = batch['labels']  # List of [N_i, 5]
            image_sizes = batch['image_sizes']
            
            # Move labels to device
            labels = [l.to(self.device) for l in labels]
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(events)
                losses = self.loss_fn(outputs, labels, image_sizes)
                loss = losses['total']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Scheduler step
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], global_step)
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Validate model."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            events = batch['events'].to(self.device)
            labels = batch['labels']
            image_sizes = batch['image_sizes']
            
            with autocast():
                outputs = self.model(events)
            
            # Convert outputs to predictions (simplified)
            # Full implementation would decode bboxes and apply NMS
            batch_preds = self._decode_predictions(outputs, image_sizes)
            
            all_predictions.extend(batch_preds)
            all_targets.extend(labels)
        
        # Compute mAP
        mAP = compute_map(all_predictions, all_targets)
        
        self.writer.add_scalar('Val/mAP', mAP, epoch)
        
        return mAP
    
    def _decode_predictions(
        self,
        outputs: dict,
        image_sizes: list
    ) -> list:
        """Decode model outputs to predictions."""
        # Simplified decoding - full YOLOX uses proper post-processing
        batch_size = outputs['p3']['cls'].shape[0]
        predictions = []
        
        for b in range(batch_size):
            preds = []
            
            for scale in ['p3', 'p4', 'p5']:
                cls = outputs[scale]['cls'][b]  # [num_classes, H, W]
                reg = outputs[scale]['reg'][b]  # [4, H, W]
                obj = outputs[scale]['obj'][b]  # [1, H, W]
                
                # Get objectness scores
                obj_scores = torch.sigmoid(obj).squeeze(0)  # [H, W]
                
                # Threshold
                mask = obj_scores > 0.5
                
                if mask.sum() > 0:
                    # Get predictions
                    ys, xs = torch.where(mask)
                    
                    for y, x in zip(ys, xs):
                        box = reg[:, y, x]
                        cls_scores = torch.sigmoid(cls[:, y, x])
                        cls_id = cls_scores.argmax()
                        score = obj_scores[y, x] * cls_scores[cls_id]
                        
                        preds.append({
                            'bbox': box.cpu().numpy(),
                            'class_id': cls_id.item(),
                            'score': score.item()
                        })
            
            predictions.append(preds)
        
        return predictions
    
    def train(self):
        """Full training loop."""
        print(f"\nStarting training for {self.config['training']['epochs']} epochs")
        print(f"Checkpoint directory: {self.checkpoint_dir}\n")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_mAP = self.validate(epoch)
            
            print(f"\nEpoch {epoch+1}: Loss={train_loss:.4f}, mAP={val_mAP:.4f}")
            
            # Save checkpoint
            is_best = val_mAP > self.best_map
            if is_best:
                self.best_map = val_mAP
            
            self._save_checkpoint(epoch, is_best)
        
        print(f"\nTraining complete! Best mAP: {self.best_map:.4f}")
        self.writer.close()


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args.config, args.resume)
    trainer.train()
```

---

## 8. Configuration

**File: `configs/gen1_config.yaml`**

```yaml
# Gen1 Dataset Configuration
# Based on paper: Ahmed et al. (arXiv:2403.10173v4)

data:
  root_dir: '/path/to/gen1'  # UPDATE THIS
  height: 304
  width: 240
  num_time_bins: 10          # T=10 bins
  bin_duration_ms: 5.0       # 5ms per bin (50ms total)

model:
  num_classes: 2             # CORRECTED: Gen1 has 2 classes (car, pedestrian)
  in_channels: 2             # Polarity channels
  tau_init: 2.0              # Initial PLIF time constant
  asab_kernel_size: 5        # TSDC kernel size (paper: "kernel size for Φ_tsdc is set to 5")

training:
  epochs: 50                 # Paper: "trained end-to-end for 50 epochs on Gen 1"
  batch_size: 24             # Paper: "batch size of 24"
  learning_rate: 2.0e-4      # Paper: "learning rate of 2×10^-4"
  weight_decay: 1.0e-4
  num_workers: 8
  checkpoint_dir: './checkpoints'
  
  # OneCycle scheduler (paper: "OneCycle learning rate schedule")
  scheduler:
    pct_start: 0.3           # Warmup percentage
    div_factor: 25           # Initial lr = max_lr / div_factor
    final_div_factor: 10000  # Final lr = max_lr / final_div_factor

# Expected results (from paper Table 1-2):
# - mAP(.5:.05:.95): 0.35
# - mAP(.5): 0.61
# - Parameters: 6.6M
# - Training time: ~8 hours on 4x 3090 (~3-4 hours on A100)
```

---

## 9. Expected Results

### Paper Reported Results (Table 1-2):

| Metric | Value |
|--------|-------|
| mAP(.5:.05:.95) | **0.35** |
| mAP(.5) | **0.61** |
| Parameters | **6.6M** |
| Training Time | ~8 hours (4× 3090) |

### Ablation Study Results (Table 4):

| Model Variant | mAP(.5) | mAP |
|---------------|---------|-----|
| Without Temporal Attention (Φ_ta) | 0.57 | 0.33 |
| Without Deformable Conv | 0.59 | 0.34 |
| Without ERS Attention | 0.59 | 0.34 |
| Without ASAB (simple accumulation) | 0.53 | 0.30 |
| **Full Model (Proposed)** | **0.61** | **0.35** |

### Key Takeaways:

1. **ASAB is critical**: Removing it drops mAP from 0.35 to 0.30
2. **Deformable convolution helps**: Standard conv drops mAP by 0.01
3. **Both attention modules matter**: SAT and ERS each contribute

---

## Quick Start Commands

```bash
# 1. Setup environment with UV
curl -LsSf https://astral.sh/uv/install.sh | sh
cd hybrid_snn_ann
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .

# 2. Download Gen1 dataset
# (Follow Prophesee instructions or use provided script)
bash scripts/download_gen1.sh

# 3. Update config with your data path
nano configs/gen1_config.yaml

# 4. Start training
python src/train.py --config configs/gen1_config.yaml

# 5. Monitor training
tensorboard --logdir checkpoints/
```

---

## References

- **Paper**: Ahmed et al., "Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention", arXiv:2403.10173v4
- **Gen1 Dataset**: [Prophesee Gen1 Automotive Detection Dataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)
- **SpikingJelly**: [SpikingJelly Documentation](https://spikingjelly.readthedocs.io/)
- **YOLOX**: [YOLOX: Exceeding YOLO Series in 2021](https://github.com/Megvii-BaseDetection/YOLOX)
