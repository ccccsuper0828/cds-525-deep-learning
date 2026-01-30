"""
HMS训练器模块
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Tuple, List, Callable
import numpy as np
from tqdm import tqdm
import logging
import json
from datetime import datetime

from .losses import HMSLoss, create_loss_function

# 尝试导入可视化模块
try:
    from ..utils.visualization import TrainingVisualizer
    VIS_AVAILABLE = True
except ImportError:
    VIS_AVAILABLE = False


class HMSTrainer:
    """
    HMS模型训练器

    Features:
    - 两阶段训练策略
    - 混合精度训练
    - 梯度累积
    - 早停
    - 学习率调度
    - 自动可视化【新增】
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda',
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 100,
        experiment_name: str = "hms_experiment",
        save_dir: str = "outputs"
    ):
        """
        Args:
            model: HMS模型
            optimizer: 优化器
            scheduler: 学习率调度器
            criterion: 损失函数（默认使用HMSLoss）
            device: 设备
            use_amp: 是否使用混合精度
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 梯度裁剪阈值
            log_interval: 日志打印间隔
            experiment_name: 实验名称【新增】
            save_dir: 保存目录【新增】
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or HMSLoss()
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.experiment_name = experiment_name
        self.save_dir = save_dir

        # 混合精度
        self.scaler = GradScaler() if self.use_amp else None

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # 日志
        self.logger = logging.getLogger(__name__)

        # 可视化器【新增】
        os.makedirs(save_dir, exist_ok=True)
        if VIS_AVAILABLE:
            self.visualizer = TrainingVisualizer(save_dir, experiment_name)
        else:
            self.visualizer = None
            print("Warning: Visualization module not available")

        # 所有预测结果存储【新增】
        self.all_val_preds = None
        self.all_val_labels = None

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        stage: str = 'full'
    ) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            stage: 训练阶段 ('stage1', 'stage2', 'full')

        Returns:
            训练指标字典
        """
        self.model.train()
        self.current_epoch = epoch

        total_loss = 0
        total_kl = 0
        total_contrastive = 0
        total_cross_modal = 0
        total_soft_label = 0
        n_batches = 0
        fusion_weights_sum = None

        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch} ({stage})')

        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            eeg = batch['eeg'].to(self.device)
            spec = batch['spec'].to(self.device)
            label = batch['label'].to(self.device)
            eeg_spec = batch.get('eeg_spec')
            if eeg_spec is not None:
                eeg_spec = eeg_spec.to(self.device)

            # 前向传播
            with autocast(enabled=self.use_amp):
                output = self.model(eeg, spec, eeg_spec, return_features=True)

                # 计算损失（包含跨模态对比学习）
                collapse_loss = self.model.get_collapse_loss() if hasattr(self.model, 'get_collapse_loss') else None
                losses = self.criterion(
                    pred=output['probs'],
                    target=label,
                    contrastive_features=output.get('contrastive_features'),
                    eeg_features=output.get('eeg_features'),      # 跨模态对比学习
                    spec_features=output.get('spec_features'),    # 跨模态对比学习
                    collapse_loss=collapse_loss
                )

                loss = losses['total'] / self.gradient_accumulation_steps

            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            # 统计
            total_loss += losses['total'].item()
            total_kl += losses.get('kl', torch.tensor(0)).item()
            total_contrastive += losses.get('contrastive', torch.tensor(0)).item()
            total_cross_modal += losses.get('cross_modal', torch.tensor(0)).item()
            total_soft_label += losses.get('soft_label', torch.tensor(0)).item()
            n_batches += 1

            # 记录融合权重
            if 'fusion_weights' in output and output['fusion_weights'] is not None:
                weights = output['fusion_weights'].mean(dim=0).detach().cpu().numpy()
                if fusion_weights_sum is None:
                    fusion_weights_sum = weights
                else:
                    fusion_weights_sum += weights

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses["total"].item():.4f}',
                'kl': f'{losses.get("kl", torch.tensor(0)).item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        metrics = {
            'train_loss': total_loss / n_batches,
            'train_kl': total_kl / n_batches,
            'train_contrastive': total_contrastive / n_batches,
            'train_cross_modal': total_cross_modal / n_batches,
            'train_soft_label': total_soft_label / n_batches
        }

        # 记录平均融合权重
        if fusion_weights_sum is not None:
            metrics['fusion_weights'] = fusion_weights_sum / n_batches

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        save_predictions: bool = False
    ) -> Dict[str, float]:
        """
        验证

        Args:
            val_loader: 验证数据加载器
            save_predictions: 是否保存预测结果

        Returns:
            验证指标字典
        """
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(val_loader, desc='Validation')

        for batch in pbar:
            eeg = batch['eeg'].to(self.device)
            spec = batch['spec'].to(self.device)
            label = batch['label'].to(self.device)
            eeg_spec = batch.get('eeg_spec')
            if eeg_spec is not None:
                eeg_spec = eeg_spec.to(self.device)

            output = self.model(eeg, spec, eeg_spec, return_features=False)

            # 计算KL散度
            kl_loss = F.kl_div(
                output['probs'].log(),
                label,
                reduction='batchmean'
            )

            total_loss += kl_loss.item()
            all_preds.append(output['probs'].cpu())
            all_labels.append(label.cpu())

        # 计算整体KL散度
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        overall_kl = F.kl_div(
            all_preds.log(),
            all_labels,
            reduction='batchmean'
        ).item()

        # 保存预测结果
        if save_predictions:
            self.all_val_preds = all_preds.numpy()
            self.all_val_labels = all_labels.numpy()

        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_kl': overall_kl
        }

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        save_dir: str,
        early_stopping_patience: int = 5,
        stage: str = 'full'
    ) -> Dict[str, List[float]]:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            n_epochs: 训练轮数
            save_dir: 模型保存目录
            early_stopping_patience: 早停耐心值
            stage: 训练阶段

        Returns:
            训练历史
        """
        os.makedirs(save_dir, exist_ok=True)

        history = {
            'train_loss': [],
            'train_kl': [],
            'val_loss': [],
            'val_kl': []
        }

        patience_counter = 0

        for epoch in range(n_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch, stage)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_kl'].append(train_metrics['train_kl'])

            # 验证
            val_metrics = self.validate(val_loader, save_predictions=(epoch == n_epochs - 1))
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_kl'].append(val_metrics['val_kl'])

            # 当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 更新可视化器【新增】
            if self.visualizer is not None:
                all_metrics = {**train_metrics, **val_metrics}
                self.visualizer.update(epoch, all_metrics, current_lr)
                if 'fusion_weights' in train_metrics:
                    self.visualizer.update_fusion_weights(train_metrics['fusion_weights'])

            # 打印
            log_msg = (
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['train_loss']:.4f}, "
                f"Val Loss={val_metrics['val_loss']:.4f}, "
                f"Val KL={val_metrics['val_kl']:.4f}, "
                f"LR={current_lr:.2e}"
            )
            self.logger.info(log_msg)
            print(log_msg)

            # 保存最佳模型
            if val_metrics['val_kl'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_kl']
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch,
                    val_metrics
                )
                patience_counter = 0
                print(f"  -> Saved best model with Val KL={val_metrics['val_kl']:.4f}")
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                print(f"Early stopping at epoch {epoch}")
                break

            # 保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth'),
                    epoch,
                    val_metrics
                )

        # 训练结束后生成所有图表【新增】
        if self.visualizer is not None:
            print("\nGenerating training visualizations...")
            self.visualizer.plot_all()

            # 如果有预测结果，生成预测分析图
            if self.all_val_preds is not None and self.all_val_labels is not None:
                self.visualizer.plot_predictions(self.all_val_labels, self.all_val_preds)

            print(f"Figures saved to: {self.visualizer.figures_dir}")

        return history

    def two_stage_training(
        self,
        train_df,
        val_df,
        dataset_class,
        dataset_kwargs: Dict,
        stage1_epochs: int = 5,
        stage2_epochs: int = 15,
        high_conf_threshold: float = 0.7,
        save_dir: str = 'models',
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict[str, List[float]]:
        """
        两阶段训练策略

        Stage 1: 高置信度样本预训练
        Stage 2: 全数据微调

        Args:
            train_df: 训练数据DataFrame
            val_df: 验证数据DataFrame
            dataset_class: Dataset类
            dataset_kwargs: Dataset参数
            stage1_epochs: 第一阶段轮数
            stage2_epochs: 第二阶段轮数
            high_conf_threshold: 高置信度阈值
            save_dir: 保存目录
            batch_size: 批大小
            num_workers: 数据加载线程数

        Returns:
            训练历史
        """
        os.makedirs(save_dir, exist_ok=True)

        # ============ Stage 1: 高置信度样本预训练 ============
        print("="*60)
        print("Stage 1: 高置信度样本预训练")
        print("="*60)

        # 筛选高置信度样本
        vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        train_df = train_df.copy()
        votes = train_df[vote_cols].values
        total_votes = votes.sum(axis=1, keepdims=True)
        max_ratio = votes.max(axis=1) / (total_votes.flatten() + 1e-8)
        high_conf_mask = max_ratio > high_conf_threshold

        high_conf_df = train_df[high_conf_mask].reset_index(drop=True)
        print(f"高置信度样本: {len(high_conf_df)} / {len(train_df)} ({100*len(high_conf_df)/len(train_df):.1f}%)")

        # 创建数据集
        high_conf_dataset = dataset_class(high_conf_df, **dataset_kwargs, mode='train')
        val_dataset = dataset_class(val_df, **dataset_kwargs, mode='val')

        high_conf_loader = DataLoader(
            high_conf_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # Stage 1 训练
        history_s1 = self.fit(
            high_conf_loader,
            val_loader,
            n_epochs=stage1_epochs,
            save_dir=os.path.join(save_dir, 'stage1'),
            early_stopping_patience=stage1_epochs,  # 不早停
            stage='stage1'
        )

        # ============ Stage 2: 全数据微调 ============
        print("\n" + "="*60)
        print("Stage 2: 全数据微调")
        print("="*60)

        # 降低学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

        # 创建全数据集
        full_dataset = dataset_class(train_df, **dataset_kwargs, mode='train')
        full_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        # Stage 2 训练
        history_s2 = self.fit(
            full_loader,
            val_loader,
            n_epochs=stage2_epochs,
            save_dir=os.path.join(save_dir, 'stage2'),
            early_stopping_patience=5,
            stage='stage2'
        )

        # 合并历史
        history = {
            'stage1': history_s1,
            'stage2': history_s2
        }

        # 保存完整训练历史【新增】
        history_path = os.path.join(save_dir, 'full_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved to: {history_path}")

        return history

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.current_epoch = checkpoint.get('epoch', 0)

        return checkpoint.get('metrics', {})


@torch.no_grad()
def inference(
    model: nn.Module,
    loader: DataLoader,
    device: str = 'cuda',
    use_tta: bool = True,
    n_tta: int = 3
) -> torch.Tensor:
    """
    推理

    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
        use_tta: 是否使用TTA
        n_tta: TTA次数

    Returns:
        (N, C) 预测概率
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

        # 原始预测
        output = model(eeg, spec, eeg_spec)
        preds.append(output['probs'])

        if use_tta:
            # TTA: 水平翻转频谱图
            output = model(eeg, torch.flip(spec, dims=[-1]), eeg_spec)
            preds.append(output['probs'])

            # TTA: 时间翻转EEG
            output = model(torch.flip(eeg, dims=[-1]), spec, eeg_spec)
            preds.append(output['probs'])

        # 平均
        pred = torch.stack(preds).mean(dim=0)
        all_preds.append(pred.cpu())

    return torch.cat(all_preds, dim=0)
