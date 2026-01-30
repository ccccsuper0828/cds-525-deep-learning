#!/usr/bin/env python
"""
HMS 推理脚本

用于生成Kaggle提交文件

使用方法:
    python scripts/inference.py \
        --checkpoint outputs/xxx/checkpoints/best_model.pth \
        --config configs/config.yaml \
        --output submission.csv
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import HMSDataset
from src.models import create_model
from src.training import inference
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='HMS Inference Script')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='Path to test.csv')
    parser.add_argument('--test_eeg_dir', type=str, default=None,
                       help='Path to test EEG directory')
    parser.add_argument('--test_spec_dir', type=str, default=None,
                       help='Path to test spectrogram directory')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output submission file path')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--use_tta', action='store_true', default=True,
                       help='Use test time augmentation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置数据路径
    data_dir = os.path.dirname(config['data']['train_csv'])
    test_csv = args.test_csv or os.path.join(data_dir, 'test.csv')
    test_eeg_dir = args.test_eeg_dir or os.path.join(data_dir, 'test_eegs')
    test_spec_dir = args.test_spec_dir or os.path.join(data_dir, 'test_spectrograms')

    print("="*60)
    print("HMS Inference")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test CSV: {test_csv}")
    print(f"Output: {args.output}")
    print(f"Use TTA: {args.use_tta}")
    print("="*60)

    # 检查文件
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not os.path.exists(test_csv):
        print(f"Error: Test CSV not found: {test_csv}")
        sys.exit(1)

    # 设置设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'

    print(f"\nUsing device: {device}")

    # 加载测试数据
    print("\nLoading test data...")
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")

    # 为测试数据添加伪标签列（推理时不需要真实标签）
    label_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    for col in label_cols:
        if col not in test_df.columns:
            test_df[col] = 1  # 伪标签

    # 创建数据集
    test_dataset = HMSDataset(
        df=test_df,
        eeg_dir=test_eeg_dir,
        spec_dir=test_spec_dir,
        eeg_length=config['data'].get('eeg_length', 2000),
        mode='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    print("\nCreating model...")
    model_config = config['model']
    model = create_model(
        model_type='full' if model_config['name'] == 'hms_full' else 'lite',
        num_classes=model_config['num_classes'],
        dim=model_config['eeg_encoder'].get('hidden_dim', 256),
        eeg_encoder_type='mamba' if model_config['eeg_encoder']['type'] == 'eeg_mamba' else 'wavenet',
        spec_encoder_type='cmfvit' if model_config['spec_encoder']['type'] == 'cmf_vit' else 'efficientnet',
        fusion_type='adaptive' if model_config['fusion']['type'] == 'adaptive_gated' else 'gated',
        use_eeg_spec=True,
        use_contrastive=False,  # 推理时不需要对比学习
        dropout=0.0  # 推理时关闭dropout
    )

    # 加载权重
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Best val KL: {checkpoint.get('best_val_loss', 'N/A')}")

    # 推理
    print("\nRunning inference...")
    predictions = inference(
        model=model,
        loader=test_loader,
        device=device,
        use_tta=args.use_tta
    )

    predictions = predictions.numpy()
    print(f"Predictions shape: {predictions.shape}")

    # 创建提交文件
    print("\nCreating submission file...")
    submission = pd.DataFrame({
        'eeg_id': test_df['eeg_id'].values,
        'seizure_vote': predictions[:, 0],
        'lpd_vote': predictions[:, 1],
        'gpd_vote': predictions[:, 2],
        'lrda_vote': predictions[:, 3],
        'grda_vote': predictions[:, 4],
        'other_vote': predictions[:, 5]
    })

    # 确保概率和为1
    prob_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    submission[prob_cols] = submission[prob_cols].div(
        submission[prob_cols].sum(axis=1), axis=0
    )

    # 保存
    submission.to_csv(args.output, index=False)
    print(f"\n✓ Submission saved to: {args.output}")

    # 显示预测分布
    print("\nPrediction statistics:")
    print("-" * 40)
    for col in prob_cols:
        print(f"{col:15s}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}")

    print("\n" + "="*60)
    print("Inference completed!")
    print("="*60)


if __name__ == "__main__":
    main()
