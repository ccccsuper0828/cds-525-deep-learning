"""
HMS Training Module
"""

from .losses import (
    KLDivergenceLoss,
    SupervisedContrastiveLoss,
    CrossModalContrastiveLoss,
    SoftLabelContrastiveLoss,
    MultiModalContrastiveLoss,
    LabelSmoothingLoss,
    FocalLoss,
    HMSLoss,
    create_loss_function
)

from .trainer import (
    HMSTrainer,
    inference
)

__all__ = [
    # Losses
    'KLDivergenceLoss',
    'SupervisedContrastiveLoss',
    'CrossModalContrastiveLoss',
    'SoftLabelContrastiveLoss',
    'MultiModalContrastiveLoss',
    'LabelSmoothingLoss',
    'FocalLoss',
    'HMSLoss',
    'create_loss_function',

    # Trainer
    'HMSTrainer',
    'inference'
]
