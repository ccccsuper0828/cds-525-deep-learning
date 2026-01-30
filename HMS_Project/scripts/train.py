#!/usr/bin/env python
"""
HMS 完整训练脚本

使用方法:
    python scripts/train.py --config configs/config.yaml

服务器运行（后台）:
    nohup python scripts/train.py --config configs/config.yaml > train.log 2>&1 &

查看训练日志:
    tail -f train.log

查看GPU使用:
    watch -n 1 nvidia-smi
"""

import os
import sys
import argparse
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import HMSDataset, create_data_loaders
from src.models import HMSModel, create_model
from src.training import HMSTrainer, HMSLoss, create_loss_function
from src.utils.visualization import TrainingVisualizer


def setup_logging(output_dir: str, experiment_name: str):
    """设置日志"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """创建优化器"""
    opt_config = config['optimizer']

    # 使用层级学习率衰减
    if opt_config.get('use_llrd', False):
        # 对预训练backbone使用较小学习率
        params = []
        base_lr = opt_config['lr']
        llrd_rate = opt_config.get('llrd_rate', 0.95)

        # 分组参数
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if 'encoder' in name and 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        params = [
            {'params': backbone_params, 'lr': base_lr * llrd_rate},
            {'params': head_params, 'lr': base_lr}
        ]
    else:
        params = model.parameters()

    if opt_config['name'].lower() == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.01),
            betas=tuple(opt_config.get('betas', [0.9, 0.999])),
            eps=opt_config.get('eps', 1e-8)
        )
    elif opt_config['name'].lower() == 'adam':
        optimizer = optim.Adam(
            params,
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0),
            betas=tuple(opt_config.get('betas', [0.9, 0.999]))
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['name']}")

    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: dict, num_training_steps: int):
    """创建学习率调度器"""
    sched_config = config['scheduler']

    if sched_config['name'] == 'cosine_warmup':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

        warmup_steps = int(sched_config.get('warmup_epochs', 2) * num_training_steps / config['training']['full']['epochs'])
        min_lr = sched_config.get('min_lr', 1e-6)
        max_lr = config['optimizer']['lr']

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
                return min_lr / max_lr + (1 - min_lr / max_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
    elif sched_config['name'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 10),
            gamma=sched_config.get('gamma', 0.1)
        )
    elif sched_config['name'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
    else:
        scheduler = None

    return scheduler


def main():
    parser = argparse.ArgumentParser(description='HMS Training Script')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--fold', type=int, default=None,
                       help='Fold number (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (use small subset)')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 覆盖配置
    if args.fold is not None:
        config['data']['fold'] = args.fold
    if args.output_dir is not None:
        config['experiment']['output_dir'] = args.output_dir

    # 创建输出目录
    output_dir = config['experiment']['output_dir']
    experiment_name = config['experiment']['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 设置日志
    logger = setup_logging(run_dir, experiment_name)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Output directory: {run_dir}")

    # 保存配置
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to: {config_save_path}")

    # 设置设备
    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    logger.info(f"Using device: {device}")

    # 设置随机种子
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # 加载数据
    logger.info("Loading data...")

    # 检查数据路径
    train_csv = config['data']['train_csv']
    if not os.path.exists(train_csv):
        logger.error(f"Training CSV not found: {train_csv}")
        logger.error("Please run: python scripts/download_data.py --data_dir ./data")
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    logger.info(f"Total samples: {len(train_df)}")

    # Debug模式使用小数据集
    if args.debug:
        train_df = train_df.head(1000)
        logger.info(f"Debug mode: using {len(train_df)} samples")

    # 数据划分
    from sklearn.model_selection import GroupKFold
    n_folds = config['data']['n_folds']
    fold = config['data']['fold']

    gkf = GroupKFold(n_splits=n_folds)
    patient_ids = train_df['patient_id'].values

    for i, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=patient_ids)):
        if i == fold:
            train_df_fold = train_df.iloc[train_idx].reset_index(drop=True)
            val_df_fold = train_df.iloc[val_idx].reset_index(drop=True)
            break

    logger.info(f"Fold {fold}: Train={len(train_df_fold)}, Val={len(val_df_fold)}")

    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        train_df=train_df_fold,
        val_df=val_df_fold,
        eeg_dir=config['data']['eeg_dir'],
        spec_dir=config['data']['spec_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )

    # 创建模型
    logger.info("Creating model...")
    model_config = config['model']
    model = create_model(
        model_type='full' if model_config['name'] == 'hms_full' else 'lite',
        num_classes=model_config['num_classes'],
        dim=model_config['eeg_encoder'].get('hidden_dim', 256),
        eeg_encoder_type='mamba' if model_config['eeg_encoder']['type'] == 'eeg_mamba' else 'wavenet',
        spec_encoder_type='cmfvit' if model_config['spec_encoder']['type'] == 'cmf_vit' else 'efficientnet',
        fusion_type='adaptive' if model_config['fusion']['type'] == 'adaptive_gated' else 'gated',
        use_eeg_spec=True,
        use_contrastive=model_config.get('use_contrastive', True),
        dropout=model_config.get('classifier_dropout', 0.3)
    )

    # 打印模型信息
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} ({n_trainable:,} trainable)")

    # 创建优化器和调度器
    optimizer = create_optimizer(model, config)

    num_training_steps = len(train_loader) * config['training']['full']['epochs']
    scheduler = create_scheduler(optimizer, config, num_training_steps)

    # 创建损失函数
    loss_config = config['loss']
    criterion = create_loss_function(
        loss_type=loss_config['type'],
        use_kl=loss_config.get('use_kl', True),
        use_contrastive=loss_config.get('use_contrastive', True),
        use_cross_modal=loss_config.get('use_cross_modal', True),
        use_soft_label_contrastive=loss_config.get('use_soft_label_contrastive', True),
        use_collapse=loss_config.get('use_collapse', True),
        kl_weight=loss_config.get('kl_weight', 1.0),
        contrastive_weight=loss_config.get('contrastive_weight', 0.1),
        cross_modal_weight=loss_config.get('cross_modal_weight', 0.05),
        soft_label_weight=loss_config.get('soft_label_weight', 0.05),
        collapse_weight=loss_config.get('collapse_weight', 0.01),
        temperature=loss_config.get('temperature', 0.07)
    )

    # 创建训练器
    trainer = HMSTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        use_amp=config['training'].get('use_amp', True),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        experiment_name=experiment_name,
        save_dir=run_dir
    )

    # 恢复训练
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 开始训练
    logger.info("Starting training...")
    print("\n" + "="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Output Dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {config['training']['full']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print("="*60 + "\n")

    training_strategy = config['training'].get('strategy', 'full')

    if training_strategy == 'two_stage':
        # 两阶段训练
        logger.info("Using two-stage training strategy")
        history = trainer.two_stage_training(
            train_df=train_df_fold,
            val_df=val_df_fold,
            dataset_class=HMSDataset,
            dataset_kwargs={
                'eeg_dir': config['data']['eeg_dir'],
                'spec_dir': config['data']['spec_dir'],
                'eeg_length': config['data'].get('eeg_length', 2000)
            },
            stage1_epochs=config['training']['two_stage']['stage1_epochs'],
            stage2_epochs=config['training']['two_stage']['stage2_epochs'],
            high_conf_threshold=config['training']['two_stage'].get('high_conf_threshold', 0.7),
            save_dir=os.path.join(run_dir, 'checkpoints'),
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers']
        )
    else:
        # 完整训练
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=config['training']['full']['epochs'],
            save_dir=os.path.join(run_dir, 'checkpoints'),
            early_stopping_patience=config['training']['early_stopping']['patience'],
            stage='full'
        )

    # 保存最终结果
    results = {
        'experiment_name': experiment_name,
        'fold': fold,
        'best_val_kl': trainer.best_val_loss,
        'config': config,
        'history': history
    }

    results_path = os.path.join(run_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info("Training completed!")
    logger.info(f"Best Val KL: {trainer.best_val_loss:.4f}")
    logger.info(f"Results saved to: {run_dir}")
    logger.info(f"Figures saved to: {run_dir}/figures/")
    logger.info(f"{'='*60}")

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best Val KL: {trainer.best_val_loss:.4f}")
    print(f"\nOutput files:")
    print(f"  - Checkpoints: {run_dir}/checkpoints/")
    print(f"  - Figures: {run_dir}/figures/")
    print(f"  - Logs: {run_dir}/logs/")
    print(f"  - Results: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
