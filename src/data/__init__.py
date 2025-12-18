"""
Data loading and event representation modules.

Components:
- EventRepresentation: Converts raw events to tensor format
- Gen1Dataset: PyTorch Dataset for Gen1 automotive detection
- collate_fn: Custom collate function for variable-length labels
"""

from .event_representation import EventRepresentation
from .gen1_dataset import Gen1Dataset, collate_fn, GEN1_CLASSES, GEN1_NUM_CLASSES

__all__ = [
    'EventRepresentation',
    'Gen1Dataset',
    'collate_fn',
    'GEN1_CLASSES',
    'GEN1_NUM_CLASSES',
]
