#!/usr/bin/env python3
"""
HMS Harmful Brain Activity Classification - Inference Script

Supports:
- Single model inference
- Ensemble inference
- Test Time Augmentation (TTA)
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict

# HMS modules
from src.data import HMSDataset, HMSDataProcessor
from src.models import create_model, HMSEnsemble
from src.training import inference
from src.utils import set_seed, get_device


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: dict, device: str) -> torch.nn.Module:
    """Load a trained model from checkpoint"""
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    if 'metrics' in checkpoint:
        print(f"  Checkpoint metrics: {checkpoint['metrics']}")

    return model


@torch.no_grad()
def inference_single(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    use_tta: bool = True,
    n_tta: int = 3
) -> np.ndarray:
    """
    Single model inference with optional TTA

    Args:
        model: Trained model
        loader: Test dataloader
        device: Device
        use_tta: Whether to use Test Time Augmentation
        n_tta: Number of TTA variations

    Returns:
        (N, C) prediction probabilities
    """
    model.eval()
    all_preds = []

    for batch in tqdm(loader, desc='Inference'):
        eeg = batch['eeg'].to(device)
        spec = batch['spec'].to(device)
        eeg_spec = batch.get('eeg_spec')
        if eeg_spec is not None:
            eeg_spec = eeg_spec.to(device)

        preds = []

        # Original prediction
        output = model(eeg, spec, eeg_spec)
        preds.append(output['probs'])

        if use_tta:
            # TTA 1: Horizontal flip spectrogram
            output = model(eeg, torch.flip(spec, dims=[-1]), eeg_spec)
            preds.append(output['probs'])

            # TTA 2: Time flip EEG
            output = model(torch.flip(eeg, dims=[-1]), spec, eeg_spec)
            preds.append(output['probs'])

            if n_tta > 3:
                # TTA 3: Both flips
                output = model(
                    torch.flip(eeg, dims=[-1]),
                    torch.flip(spec, dims=[-1]),
                    eeg_spec
                )
                preds.append(output['probs'])

        # Average predictions
        pred = torch.stack(preds).mean(dim=0)
        all_preds.append(pred.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


@torch.no_grad()
def inference_ensemble(
    models: List[torch.nn.Module],
    loader: DataLoader,
    device: str,
    weights: Optional[List[float]] = None,
    use_tta: bool = True
) -> np.ndarray:
    """
    Ensemble inference

    Args:
        models: List of trained models
        loader: Test dataloader
        device: Device
        weights: Optional weights for each model
        use_tta: Whether to use TTA

    Returns:
        (N, C) prediction probabilities
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

    all_model_preds = []

    for i, model in enumerate(models):
        print(f"Running inference for model {i+1}/{len(models)}")
        preds = inference_single(model, loader, device, use_tta=use_tta)
        all_model_preds.append(preds * weights[i])

    # Weighted average
    ensemble_preds = np.sum(all_model_preds, axis=0)

    return ensemble_preds


def create_submission(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    output_path: str,
    vote_cols: List[str] = None
):
    """
    Create Kaggle submission file

    Args:
        predictions: (N, C) prediction probabilities
        test_df: Test dataframe
        output_path: Output CSV path
        vote_cols: Column names for predictions
    """
    if vote_cols is None:
        vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote',
                     'lrda_vote', 'grda_vote', 'other_vote']

    # Create submission dataframe
    submission = pd.DataFrame({
        'eeg_id': test_df['eeg_id']
    })

    for i, col in enumerate(vote_cols):
        submission[col] = predictions[:, i]

    # Ensure probabilities sum to 1
    prob_cols = submission[vote_cols]
    row_sums = prob_cols.sum(axis=1)
    submission[vote_cols] = prob_cols.div(row_sums, axis=0)

    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())

    return submission


def run_inference(config: dict, args):
    """Main inference function"""
    # Setup
    set_seed(config['training'].get('seed', 42))
    device = args.device or config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = get_device()
    print(f"Using device: {device}")

    # Load test data
    data_config = config['data']
    test_csv = args.test_csv or data_config.get('test_csv', 'test.csv')
    test_df = pd.read_csv(test_csv)
    print(f"Loaded {len(test_df)} test samples")

    # Create processor
    processor = HMSDataProcessor(
        eeg_dir=args.eeg_dir or data_config.get('eeg_dir'),
        spec_dir=args.spec_dir or data_config.get('spec_dir'),
        eeg_channels=data_config.get('eeg_channels', 20),
        eeg_length=data_config.get('eeg_length', 2000)
    )

    # Create test dataset
    test_dataset = HMSDataset(
        df=test_df,
        processor=processor,
        vote_cols=data_config.get('vote_cols'),
        mode='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size or config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    # Inference settings
    inf_config = config.get('inference', {})
    use_tta = args.tta if args.tta is not None else inf_config.get('use_tta', True)
    n_tta = inf_config.get('n_tta', 3)

    # Load model(s)
    if args.ensemble or inf_config.get('use_ensemble', False):
        # Ensemble inference
        model_paths = args.model_paths or inf_config.get('model_paths', [])
        if not model_paths:
            raise ValueError("No model paths provided for ensemble")

        models = []
        for path in model_paths:
            model = load_model(path, config, device)
            models.append(model)

        weights = inf_config.get('ensemble_weights')
        predictions = inference_ensemble(models, test_loader, device, weights, use_tta)

    else:
        # Single model inference
        model_path = args.model_path or os.path.join(
            config.get('experiment', {}).get('checkpoint_dir', 'checkpoints'),
            'best_model.pth'
        )
        model = load_model(model_path, config, device)
        predictions = inference_single(model, test_loader, device, use_tta, n_tta)

    # Create submission
    output_dir = args.output_dir or inf_config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'submission.csv')

    submission = create_submission(
        predictions,
        test_df,
        output_path,
        vote_cols=data_config.get('vote_cols')
    )

    # Optionally save raw predictions
    if inf_config.get('save_predictions', True):
        np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
        print(f"Raw predictions saved to {output_dir}/predictions.npy")

    return submission


def main():
    parser = argparse.ArgumentParser(description='HMS Inference Script')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--model_paths', type=str, nargs='+', default=None,
                       help='Paths to model checkpoints (for ensemble)')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='Path to test CSV')
    parser.add_argument('--eeg_dir', type=str, default=None,
                       help='Path to EEG directory')
    parser.add_argument('--spec_dir', type=str, default=None,
                       help='Path to spectrogram directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--tta', type=bool, default=None,
                       help='Use TTA')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble inference')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Run inference
    run_inference(config, args)


if __name__ == '__main__':
    main()
