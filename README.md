# Hybrid SNN-ANN for Event-Based Object Detection

Implementation of "Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention" by Ahmed et al. (arXiv:2403.10173v4)

## Architecture Overview

```
Events [T,B,2,H,W] → SNN Backbone → ASAB Bridge → ANN Backbone → YOLOX Head → Detections
```

### Key Components

| Component | Description | Output |
|-----------|-------------|--------|
| **SNN Backbone** | 4 PLIF conv blocks (64→128→256→256) | T×256×H/8×W/8 |
| **ASAB Module** | SAT + ERS attention bridge | 256×H/8×W/8 |
| **ANN Backbone** | 4 conv blocks with multi-scale output | P3, P4, P5 |
| **YOLOX Head** | Decoupled detection head | boxes, scores, labels |

### Critical Corrections from Paper

This implementation corrects several issues found in other implementations:

| Issue | Wrong | Correct (Paper Table 9) |
|-------|-------|-------------------------|
| PLIF tau | τ = sigmoid(w) | τ = 1/sigmoid(w) |
| SNN channels | 32→64→128→256 | 64→128→256→256 |
| SNN strides | s1,s2,s1,s2 | s2,s2,s2,s1 |
| ANN strides | all s1 | s1,s2,s1,s2 |
| TSDC groups | = channels | = time steps T |
| Gen1 classes | 3 | 2 (car, pedestrian) |

## Installation

### Using UV (Recommended)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project
cd hybrid_snn_ann

# Create virtual environment
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Install PyTorch with CUDA (adjust for your CUDA version)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Dataset Setup

### Gen1 Dataset

Download the Prophesee Gen1 Automotive Detection Dataset:
- URL: https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/

Expected structure:
```
data/gen1/
├── train/
│   ├── <sequence_name>/
│   │   ├── events.dat or events.h5
│   │   └── labels_v2.npy or <sequence>_bbox.npy
│   └── ...
└── test/
    ├── <sequence_name>/
    │   ├── events.dat or events.h5
    │   └── labels_v2.npy or <sequence>_bbox.npy
    └── ...
```

Run the setup script for instructions:
```bash
bash scripts/download_gen1.sh
```

## Training

### Basic Training

```bash
python src/train.py --data_root ./data/gen1 --epochs 50 --batch_size 8
```

### Using Config File

```bash
python src/train.py --config configs/gen1_config.yaml
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `./data/gen1` | Path to Gen1 dataset |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 24 | Batch size (paper: 24 on 4×3090) |
| `--learning_rate` | 2e-4 | Learning rate |
| `--num_time_bins` | 10 | Number of time bins T |
| `--no_amp` | False | Disable mixed precision |

### Resume Training

```bash
python src/train.py --config configs/gen1_config.yaml --resume checkpoints/latest.pth
```

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pth --data_root ./data/gen1
```

## Usage

### Quick Start

```python
from src.models import build_model
import torch

# Build model
model = build_model(num_classes=2, num_time_bins=10)

# Create dummy input (T=10 time bins, batch=2, 2 channels, 304×240)
events = torch.randn(10, 2, 2, 304, 240)

# Inference
model.eval()
predictions = model.predict(events, score_thresh=0.3)

# Training
model.train()
outputs = model(events)
```

### Component Usage

```python
from src.models import SNNBackbone, ASABModule, ANNBackbone, YOLOXHead

# SNN Backbone
snn = SNNBackbone(in_channels=2, tau_init=2.0)
snn_out = snn(events)  # [T, B, 256, H/8, W/8]

# ASAB Bridge
asab = ASABModule(channels=256, num_time_bins=10)
bridge_out = asab(snn_out)  # [B, 256, H/8, W/8]

# ANN Backbone
ann = ANNBackbone(in_channels=256)
features = ann(bridge_out)  # {'p3': ..., 'p4': ..., 'p5': ...}

# Detection Head
head = YOLOXHead(num_classes=2)
outputs = head(features)
```

## Expected Results

From Paper Table 1-2 (Gen1 Dataset):

| Metric | Value |
|--------|-------|
| mAP(.5:.05:.95) | 0.35 |
| mAP(.5) | 0.61 |
| Parameters | 6.6M |

Ablation (Table 2):
- Without ASAB: mAP drops from 0.35 → 0.30
- Without ERS attention: mAP drops to 0.33
- Without SAT attention: mAP drops to 0.32

## Project Structure

```
hybrid_snn_ann/
├── configs/
│   └── gen1_config.yaml      # Training configuration
├── src/
│   ├── data/
│   │   ├── event_representation.py  # Event→voxel conversion
│   │   └── gen1_dataset.py          # Gen1 dataset loader
│   ├── models/
│   │   ├── plif_neuron.py    # Corrected PLIF neuron
│   │   ├── snn_backbone.py   # SNN feature extractor
│   │   ├── asab_module.py    # SAT + ERS attention bridge
│   │   ├── ann_backbone.py   # ANN with multi-scale output
│   │   ├── yolox_head.py     # Detection head
│   │   └── hybrid_model.py   # Full model
│   ├── utils/
│   │   ├── losses.py         # YOLOX loss
│   │   └── metrics.py        # mAP computation
│   └── train.py              # Training script
├── scripts/
│   ├── download_gen1.sh      # Dataset download helper
│   └── evaluate.py           # Evaluation script
├── checkpoints/              # Model checkpoints
└── pyproject.toml            # UV/pip dependencies
```

## Key Implementation Details

### PLIF Neuron (Equation 2)
```python
# Paper: V[t] = V[t-1] + (1/τ)(X[t] - (V[t-1] - Vreset))
# Where τ = 1/sigmoid(w), NOT sigmoid(w)!
tau = 1.0 / torch.sigmoid(self.w)
v_new = v_mem + (1.0 / tau) * (x - (v_mem - self.v_reset))
```

### TSDC Groups
```python
# Paper: "set group of deformable convolution kernels equal to time steps T"
# NOT channels as in some implementations
groups = num_time_bins  # T groups, not C groups
```

### Multi-Scale Output
```python
# ANN outputs at 3 scales for detection
# P3: H/8 × W/8   (38×30 for Gen1)
# P4: H/16 × W/16 (19×15 for Gen1)
# P5: H/32 × W/32 (10×8 for Gen1)
```

## Testing Individual Components

```bash
# Test PLIF neuron
python src/models/plif_neuron.py

# Test SNN backbone
python src/models/snn_backbone.py

# Test ASAB module
python src/models/asab_module.py

# Test ANN backbone
python src/models/ann_backbone.py

# Test YOLOX head
python src/models/yolox_head.py

# Test full model
python src/models/hybrid_model.py
```

## Citation

```bibtex
@article{ahmed2024efficient,
  title={Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention},
  author={Ahmed, Sheikh Shams Azam and Sinha, Saurabh and others},
  journal={arXiv preprint arXiv:2403.10173},
  year={2024}
}
```

## License

This implementation is for research purposes. Please refer to the original paper for licensing of the architecture design.
