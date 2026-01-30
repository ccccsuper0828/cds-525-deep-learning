#!/usr/bin/env python3
"""
HMS Harmful Brain Activity Classification - Training Script

2025 SOTA Implementation:
- EEGMamba (Bidirectional SSM + MoE) for EEG encoding
- CMFViT (CNN + ViT fusion) for Spectrogram encoding
- Adaptive Gated Fusion with anti-modality collapse
- Two-stage training strategy (Kaggle 1st place)
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import pandas as pd
import numpy as np
from pathlib import Path

# HMS modules
from src.data import HMSDataset, HMSDataProcessor, EEGAugmentation, SpectrogramAugmentation
from src.models import create_model, HMSModel
from src.training import HMSTrainer, HMSLoss, create_loss_function
from src.utils import (
    set_seed,
    setup_logging,
    create_leak_free_splits,
    create_patient_balanced_sampler,
    get_device,
    print_model_summary
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer with optional layer-wise learning rate decay"""
    opt_config = config['optimizer']

    # Separate parameters for pretrained backbones and new layers
    pretrained_params = []
    new_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'backbone' in name or 'encoder' in name:
            pretrained_params.append(param)
        else:
            new_params.append(param)

    if opt_config.get('use_llrd', False):
        # Layer-wise learning rate decay
        param_groups = [
            {'params': pretrained_params, 'lr': opt_config['lr'] * opt_config.get('llrd_rate', 0.95)},
            {'params': new_params, 'lr': opt_config['lr']}
        ]
    else:
        param_groups = [
            {'params': pretrained_params, 'lr': opt_config['lr']},
            {'params': new_params, 'lr': opt_config['lr']}
        ]

    if opt_config['name'] == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.01),
            betas=tuple(opt_config.get('betas', [0.9, 0.999])),
            eps=opt_config.get('eps', 1e-8)
        )
    elif opt_config['name'] == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0),
            betas=tuple(opt_config.get('betas', [0.9, 0.999]))
        )
    elif opt_config['name'] == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=opt_config['lr'],
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['name']}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict, n_epochs: int, steps_per_epoch: int):
    """Create learning rate scheduler"""
    sched_config = config['scheduler']
    warmup_epochs = sched_config.get('warmup_epochs', 0)
    total_steps = n_epochs * steps_per_epoch

    if sched_config['name'] == 'cosine_warmup':
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            warmup_steps = warmup_epochs * steps_per_epoch
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(sched_config.get('min_lr', 1e-6) / config['optimizer']['lr'],
                      0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

    elif sched_config['name'] == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=sched_config.get('min_lr', 1e-6)
        )

    elif sched_config['name'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 10) * steps_per_epoch,
            gamma=sched_config.get('gamma', 0.1)
        )

    elif sched_config['name'] == 'reduce_on_plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_config.get('factor', 0.5),
            patience=sched_config.get('patience', 5),
            min_lr=sched_config.get('min_lr', 1e-6)
        )

    else:
        scheduler = None

    return scheduler


def prepare_data(config: dict):
    """Prepare training and validation data"""
    data_config = config['data']

    # Load CSV
    train_df = pd.read_csv(data_config['train_csv'])
    print(f"Loaded {len(train_df)} samples")

    # Create data processor
    processor = HMSDataProcessor(
        eeg_dir=data_config['eeg_dir'],
        spec_dir=data_config['spec_dir'],
        eeg_channels=data_config.get('eeg_channels', 20),
        eeg_length=data_config.get('eeg_length', 2000)
    )

    # Create fold splits (leak-free)
    n_folds = data_config.get('n_folds', 5)
    fold = data_config.get('fold', 0)

    splits = create_leak_free_splits(
        train_df,
        n_splits=n_folds,
        patient_col='patient_id',
        spec_col='spectrogram_id'
    )

    train_idx, val_idx = splits[fold]
    train_df_fold = train_df.iloc[train_idx].reset_index(drop=True)
    val_df_fold = train_df.iloc[val_idx].reset_index(drop=True)

    print(f"Fold {fold}: Train={len(train_df_fold)}, Val={len(val_df_fold)}")

    # Check for patient leakage
    train_patients = set(train_df_fold['patient_id'])
    val_patients = set(val_df_fold['patient_id'])
    leak = train_patients & val_patients
    if len(leak) > 0:
        print(f"WARNING: Patient leakage detected! {len(leak)} patients in both sets")
    else:
        print("No patient leakage detected")

    return train_df_fold, val_df_fold, processor


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    processor: HMSDataProcessor,
    config: dict
):
    """Create training and validation dataloaders"""
    data_config = config['data']
    train_config = config['training']
    aug_config = config.get('augmentation', {})

    # Create augmentations
    eeg_aug = None
    spec_aug = None

    if aug_config.get('eeg', {}).get('enable', True):
        eeg_aug_config = aug_config.get('eeg', {})
        eeg_aug = EEGAugmentation(
            time_shift_prob=eeg_aug_config.get('time_shift_prob', 0.3),
            time_shift_max=eeg_aug_config.get('time_shift_max', 50),
            add_noise_prob=eeg_aug_config.get('add_noise_prob', 0.3),
            noise_std=eeg_aug_config.get('noise_std', 0.1),
            channel_dropout_prob=eeg_aug_config.get('channel_dropout_prob', 0.2),
            channel_dropout_ratio=eeg_aug_config.get('channel_dropout_ratio', 0.1)
        )

    if aug_config.get('spec', {}).get('enable', True):
        spec_aug_config = aug_config.get('spec', {})
        spec_aug = SpectrogramAugmentation(
            horizontal_flip_prob=spec_aug_config.get('horizontal_flip_prob', 0.5),
            freq_dropout_prob=spec_aug_config.get('freq_dropout_prob', 0.2),
            freq_dropout_width=spec_aug_config.get('freq_dropout_width', 10),
            time_dropout_prob=spec_aug_config.get('time_dropout_prob', 0.2),
            time_dropout_width=spec_aug_config.get('time_dropout_width', 20)
        )

    # Mixup config
    mixup_config = None
    if aug_config.get('mixup', {}).get('enable', False):
        mixup_config = {
            'alpha': aug_config['mixup'].get('alpha', 0.4),
            'prob': aug_config['mixup'].get('prob', 0.5)
        }

    # Create datasets
    train_dataset = HMSDataset(
        df=train_df,
        processor=processor,
        vote_cols=data_config.get('vote_cols'),
        mode='train',
        eeg_augmentation=eeg_aug,
        spec_augmentation=spec_aug,
        mixup_config=mixup_config
    )

    val_dataset = HMSDataset(
        df=val_df,
        processor=processor,
        vote_cols=data_config.get('vote_cols'),
        mode='val'
    )

    # Create sampler for training
    sampler = None
    shuffle = True
    if data_config.get('use_patient_balanced_sampler', True):
        sampler = create_patient_balanced_sampler(
            train_df,
            power=data_config.get('sampler_power', 0.5),
            patient_col='patient_id'
        )
        shuffle = False

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=train_config.get('pin_memory', True),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=train_config.get('pin_memory', True)
    )

    return train_loader, val_loader


def train(config: dict):
    """Main training function"""
    # Setup
    set_seed(config['training'].get('seed', 42))
    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = get_device()
    print(f"Using device: {device}")

    # Create output directories
    exp_config = config.get('experiment', {})
    output_dir = exp_config.get('output_dir', 'outputs')
    checkpoint_dir = exp_config.get('checkpoint_dir', 'checkpoints')
    log_dir = exp_config.get('log_dir', 'logs')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(log_dir, name='hms_train')
    logger.info(f"Starting training with config: {config.get('experiment', {}).get('name', 'hms')}")

    # Prepare data
    train_df, val_df, processor = prepare_data(config)

    # Create model
    model_config = config['model']
    model = create_model(
        model_type=model_config.get('name', 'hms_full'),
        eeg_encoder_type=model_config.get('eeg_encoder', {}).get('type', 'eeg_mamba'),
        spec_encoder_type=model_config.get('spec_encoder', {}).get('type', 'cmf_vit'),
        fusion_type=model_config.get('fusion', {}).get('type', 'adaptive_gated'),
        num_classes=model_config.get('num_classes', 6),
        eeg_encoder_config=model_config.get('eeg_encoder', {}),
        spec_encoder_config=model_config.get('spec_encoder', {}),
        fusion_config=model_config.get('fusion', {}),
        use_contrastive=model_config.get('use_contrastive', True),
        contrastive_dim=model_config.get('contrastive_dim', 128)
    )

    print_model_summary(model)
    logger.info(f"Model created: {model_config.get('name', 'hms_full')}")

    # Training strategy
    train_config = config['training']
    strategy = train_config.get('strategy', 'full')

    if strategy == 'two_stage':
        # Two-stage training
        two_stage_config = train_config.get('two_stage', {})

        # Stage 1: High confidence samples
        stage1_epochs = two_stage_config.get('stage1_epochs', 5)
        stage2_epochs = two_stage_config.get('stage2_epochs', 20)

        # Create optimizer and scheduler for stage 1
        optimizer = create_optimizer(model, config)
        total_epochs = stage1_epochs + stage2_epochs
        steps_per_epoch = len(train_df) // train_config['batch_size']
        scheduler = create_scheduler(optimizer, config, total_epochs, steps_per_epoch)

        # Create loss function
        loss_config = config.get('loss', {})
        criterion = create_loss_function(
            loss_type=loss_config.get('type', 'hms'),
            use_kl=loss_config.get('use_kl', True),
            use_contrastive=loss_config.get('use_contrastive', True),
            use_collapse=loss_config.get('use_collapse', True),
            kl_weight=loss_config.get('kl_weight', 1.0),
            contrastive_weight=loss_config.get('contrastive_weight', 0.1),
            collapse_weight=loss_config.get('collapse_weight', 0.01),
            temperature=loss_config.get('temperature', 0.07)
        )

        # Create trainer
        trainer = HMSTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            use_amp=train_config.get('use_amp', True),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
            max_grad_norm=train_config.get('max_grad_norm', 1.0)
        )

        # Dataset kwargs for two-stage training
        dataset_kwargs = {
            'processor': processor,
            'vote_cols': config['data'].get('vote_cols')
        }

        # Run two-stage training
        history = trainer.two_stage_training(
            train_df=train_df,
            val_df=val_df,
            dataset_class=HMSDataset,
            dataset_kwargs=dataset_kwargs,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            high_conf_threshold=two_stage_config.get('high_conf_threshold', 0.7),
            save_dir=checkpoint_dir,
            batch_size=train_config['batch_size'],
            num_workers=train_config.get('num_workers', 4)
        )

    else:
        # Full training
        full_config = train_config.get('full', {})
        n_epochs = full_config.get('epochs', 30)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_df, val_df, processor, config
        )

        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config, n_epochs, len(train_loader))

        # Create loss function
        loss_config = config.get('loss', {})
        criterion = create_loss_function(
            loss_type=loss_config.get('type', 'hms'),
            use_kl=loss_config.get('use_kl', True),
            use_contrastive=loss_config.get('use_contrastive', True),
            use_collapse=loss_config.get('use_collapse', True),
            kl_weight=loss_config.get('kl_weight', 1.0),
            contrastive_weight=loss_config.get('contrastive_weight', 0.1),
            collapse_weight=loss_config.get('collapse_weight', 0.01),
            temperature=loss_config.get('temperature', 0.07)
        )

        # Create trainer
        trainer = HMSTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            use_amp=train_config.get('use_amp', True),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
            max_grad_norm=train_config.get('max_grad_norm', 1.0)
        )

        # Run training
        early_stopping = train_config.get('early_stopping', {})
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            save_dir=checkpoint_dir,
            early_stopping_patience=early_stopping.get('patience', 7)
        )

    logger.info("Training completed!")
    print("Training completed!")

    return history


def main():
    parser = argparse.ArgumentParser(description='HMS Training Script')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--fold', type=int, default=None,
                       help='Fold number (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.fold is not None:
        config['data']['fold'] = args.fold
    if args.device is not None:
        config['training']['device'] = args.device
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        if config['training'].get('strategy') == 'two_stage':
            config['training']['two_stage']['stage2_epochs'] = args.epochs
        else:
            config['training']['full']['epochs'] = args.epochs

    # Run training
    train(config)


if __name__ == '__main__':
    main()
