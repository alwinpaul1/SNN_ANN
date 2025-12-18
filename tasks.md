# Hybrid SNN-ANN Implementation Tasks

## Project Status: Core Implementation Complete ✓

---

## Phase 1: Environment Setup
- [ ] Install UV package manager
- [ ] Create virtual environment with Python 3.10
- [ ] Install PyTorch with CUDA support (cu121 or cu118)
- [ ] Install project dependencies via `uv pip install -e .`
- [ ] Verify GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Phase 2: Dataset Preparation

### Gen1 Dataset
- [ ] Download Gen1 dataset from Prophesee website
  - URL: https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/
- [ ] Extract and organize into `data/gen1/{train,val,test}/` structure
- [ ] Verify `.dat` files have corresponding `_bbox.npy` annotation files
- [ ] (Optional) Install Metavision SDK for faster event loading
  - Fallback parser available but slower

### Data Verification
- [ ] Run dataset test: `python -c "from src.data import Gen1Dataset; print('OK')"`
- [ ] Check sample counts for train/val/test splits
- [ ] Visualize a few samples to verify event→voxel conversion

---

## Phase 3: Model Verification

### Unit Tests
- [ ] Test PLIF neuron forward/backward pass
  ```bash
  python src/models/plif_neuron.py
  ```
- [ ] Test SNN backbone dimensions
  ```bash
  python src/models/snn_backbone.py
  ```
- [ ] Test ASAB module (SAT + ERS attention)
  ```bash
  python src/models/asab_module.py
  ```
- [ ] Test ANN backbone multi-scale outputs
  ```bash
  python src/models/ann_backbone.py
  ```
- [ ] Test YOLOX head
  ```bash
  python src/models/yolox_head.py
  ```
- [ ] Test full model end-to-end
  ```bash
  python src/models/hybrid_model.py
  ```

### Parameter Count Verification
- [ ] Verify total parameters ~6.6M (matching paper)
- [ ] Check per-component parameter breakdown:
  - SNN Backbone: ~1.2M
  - ASAB Module: ~0.8M
  - ANN Backbone: ~1.8M
  - YOLOX Head: ~2.8M

---

## Phase 4: Training

### Initial Training Run
- [ ] Start training with small batch size to verify pipeline
  ```bash
  python src/train.py --data_root ./data/gen1 --batch_size 4 --epochs 1
  ```
- [ ] Check loss values are reasonable (not NaN/Inf)
- [ ] Verify checkpoint saving works

### Full Training
- [ ] Run full 50-epoch training
  ```bash
  python src/train.py --config configs/gen1_config.yaml
  ```
- [ ] Monitor training with TensorBoard (if added)
- [ ] Track best validation loss checkpoint

### Training Optimizations
- [ ] Enable mixed precision training (default)
- [ ] Tune batch size for available GPU memory
- [ ] Experiment with learning rate (paper: 2e-4)

---

## Phase 5: Evaluation

### Evaluation Script
- [x] Implement `scripts/evaluate.py` for standalone evaluation
- [ ] Load best checkpoint
- [ ] Run inference on test set
- [ ] Compute mAP metrics

### Expected Results (Paper Baseline)
- [ ] Achieve mAP(.5:.95) ≈ 0.35
- [ ] Achieve mAP(.5) ≈ 0.61
- [ ] Compare per-class AP (car vs pedestrian)

### Ablation Studies
- [ ] Train without ASAB module (expect mAP drop to ~0.30)
- [ ] Train without ERS attention (expect mAP drop to ~0.33)
- [ ] Train without SAT attention (expect mAP drop to ~0.32)

---

## Phase 6: Inference & Demo

### Create Inference Script
- [ ] Implement `src/inference.py` for single-file inference
- [ ] Support both `.dat` files and pre-processed tensors
- [ ] Add visualization of detections

### Demo Notebook
- [ ] Create `notebooks/demo.ipynb` with:
  - Model loading
  - Event visualization
  - Detection visualization
  - Performance timing

---

## Phase 7: Code Quality

### Testing
- [ ] Add pytest unit tests in `tests/` directory
- [ ] Test data augmentation correctness
- [ ] Test NMS implementation
- [ ] Test mAP computation against known values

### Code Style
- [ ] Run `black` formatter
- [ ] Run `ruff` linter
- [ ] Add type hints where missing
- [ ] Improve docstrings

### Documentation
- [x] Add inline comments for complex operations
- [x] Document ASAB attention mechanisms
- [ ] Add architecture diagrams

---

## Phase 8: Extensions (Optional)

### Performance Improvements
- [ ] Implement TorchScript/ONNX export
- [ ] Profile and optimize bottlenecks
- [ ] Add gradient checkpointing for larger batches

### Additional Features
- [ ] Support for 1Mpx dataset (higher resolution)
- [ ] Real-time inference pipeline
- [ ] Webcam/event camera demo (if hardware available)

### Pretrained Weights
- [ ] Train to convergence
- [ ] Save and share pretrained weights
- [ ] Add weight loading in `build_model(pretrained=True)`

---

## Known Issues & Notes

### Architecture Corrections Applied
✓ PLIF tau: Using τ = 1/sigmoid(w) (not sigmoid(w))
✓ SNN channels: 64→128→256→256 (not 32→64→128→256)
✓ SNN strides: s2,s2,s2,s1 (not s1,s2,s1,s2)
✓ ANN strides: s1,s2,s1,s2 (not all s1)
✓ TSDC groups: = T (not = channels)
✓ Gen1 classes: 2 (not 3)

### Potential Issues to Watch
- Memory usage with T=10 time bins and batch_size=24
- Deformable convolution compatibility with different PyTorch versions
- Metavision SDK installation can be tricky

---

## Quick Reference Commands

```bash
# Setup
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Test model
python src/models/hybrid_model.py

# Train
python src/train.py --config configs/gen1_config.yaml

# Train (custom)
python src/train.py \
    --data_root /path/to/gen1 \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 2e-4

# Resume training
python src/train.py --config configs/gen1_config.yaml --resume checkpoints/latest.pth

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best.pth --data_root ./data/gen1
```

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Setup | 30 min | ⬜ Not started |
| Phase 2: Dataset | 1-2 hrs | ⬜ Not started |
| Phase 3: Verification | 1 hr | ⬜ Not started |
| Phase 4: Training | 4-8 hrs | ⬜ Not started |
| Phase 5: Evaluation | 1 hr | ⬜ Not started |
| Phase 6: Demo | 2 hrs | ⬜ Not started |
| Phase 7: Quality | 2-3 hrs | ⬜ Not started |
| Phase 8: Extensions | Variable | ⬜ Optional |

**Total estimated time: ~12-18 hours** (excluding training GPU time)

---

## Implementation Complete ✓

All core components have been implemented:

| Component | File | Status |
|-----------|------|--------|
| PLIF Neuron | `src/models/plif_neuron.py` | ✓ Complete |
| SNN Backbone | `src/models/snn_backbone.py` | ✓ Complete |
| ASAB Module | `src/models/asab_module.py` | ✓ Complete |
| ANN Backbone | `src/models/ann_backbone.py` | ✓ Complete |
| YOLOX Head | `src/models/yolox_head.py` | ✓ Complete |
| Hybrid Model | `src/models/hybrid_model.py` | ✓ Complete |
| Event Representation | `src/data/event_representation.py` | ✓ Complete |
| Gen1 Dataset | `src/data/gen1_dataset.py` | ✓ Complete |
| Loss Functions | `src/utils/losses.py` | ✓ Complete |
| Metrics | `src/utils/metrics.py` | ✓ Complete |
| Training Script | `src/train.py` | ✓ Complete |
| Evaluation Script | `scripts/evaluate.py` | ✓ Complete |
| Config | `configs/gen1_config.yaml` | ✓ Complete |
