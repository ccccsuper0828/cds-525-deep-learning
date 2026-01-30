"""
HMS Models Module
"""

from .eeg_encoder import (
    EEGMambaEncoder,
    WaveNetEncoder,
    BidirectionalMamba,
    MixtureOfExperts,
    create_eeg_encoder
)

from .spec_encoder import (
    CMFViTEncoder,
    EfficientNetEncoder,
    ViTEncoder,
    MultiScaleSpecEncoder,
    create_spec_encoder
)

from .fusion import (
    AdaptiveGatedFusion,
    GatedFusion,
    SimpleConcatFusion,
    CrossModalAttention,
    ModalityQualityEstimator,
    create_fusion_module
)

from .model import (
    HMSModel,
    HMSModelLite,
    HMSEnsemble,
    ContrastiveHead,
    ClassificationHead,
    create_model
)

__all__ = [
    # EEG Encoders
    'EEGMambaEncoder',
    'WaveNetEncoder',
    'BidirectionalMamba',
    'MixtureOfExperts',
    'create_eeg_encoder',

    # Spectrogram Encoders
    'CMFViTEncoder',
    'EfficientNetEncoder',
    'ViTEncoder',
    'MultiScaleSpecEncoder',
    'create_spec_encoder',

    # Fusion
    'AdaptiveGatedFusion',
    'GatedFusion',
    'SimpleConcatFusion',
    'CrossModalAttention',
    'ModalityQualityEstimator',
    'create_fusion_module',

    # Models
    'HMSModel',
    'HMSModelLite',
    'HMSEnsemble',
    'ContrastiveHead',
    'ClassificationHead',
    'create_model'
]
