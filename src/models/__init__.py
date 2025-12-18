"""
Model components for Hybrid SNN-ANN Detector

Components:
- PLIFNeuron: Parametric Leaky Integrate-and-Fire neuron
- SNNBackbone: SNN feature extractor (4 layers)
- ASABModule: Attention-based SNN-ANN Bridge
- ANNBackbone: ANN feature extractor (4 layers)
- YOLOXHead: Detection head
- HybridSNNANNDetector: Complete model
"""

from .plif_neuron import PLIFNeuron, PLIFNeuronMultiStep
from .snn_backbone import SNNBackbone, SNNConvBlock
from .asab_module import ASABModule, SpatialAwareTemporalAttention, EventRateSpatialAttention
from .ann_backbone import ANNBackbone, ANNConvBlock
from .yolox_head import YOLOXHead
from .hybrid_model import HybridSNNANNDetector, HybridBackbone, build_model

__all__ = [
    'PLIFNeuron',
    'PLIFNeuronMultiStep',
    'SNNBackbone',
    'SNNConvBlock',
    'ASABModule',
    'SpatialAwareTemporalAttention',
    'EventRateSpatialAttention',
    'ANNBackbone',
    'ANNConvBlock',
    'YOLOXHead',
    'HybridSNNANNDetector',
    'HybridBackbone',
    'build_model',
]
