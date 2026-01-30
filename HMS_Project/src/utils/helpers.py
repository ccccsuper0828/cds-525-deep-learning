"""
HMS工具函数
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from typing import List, Tuple, Optional, Dict
import logging


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, name: str = 'hms'):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 文件处理器
    fh = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    fh.setLevel(logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def compute_patient_weights(
    df: pd.DataFrame,
    power: float = 0.5,
    patient_col: str = 'patient_id'
) -> np.ndarray:
    """
    计算患者平衡采样权重

    Args:
        df: DataFrame
        power: 权重指数 (0.5 = sqrt平衡)
        patient_col: 患者ID列名

    Returns:
        每个样本的权重
    """
    patient_counts = df.groupby(patient_col).size()
    max_count = patient_counts.max()

    weights = df[patient_col].map(
        lambda x: (max_count / patient_counts[x]) ** power
    ).values

    return weights.astype(np.float32)


def create_patient_balanced_sampler(
    df: pd.DataFrame,
    power: float = 0.5,
    patient_col: str = 'patient_id'
) -> WeightedRandomSampler:
    """
    创建患者平衡采样器

    Args:
        df: DataFrame
        power: 权重指数
        patient_col: 患者ID列名

    Returns:
        WeightedRandomSampler
    """
    weights = compute_patient_weights(df, power, patient_col)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(df),
        replacement=True
    )

    return sampler


def create_group_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = 'patient_id',
    stratify_col: Optional[str] = 'expert_consensus'
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    创建GroupKFold划分

    Args:
        df: DataFrame
        n_splits: 折数
        group_col: 分组列
        stratify_col: 分层列（可选）

    Returns:
        List of (train_idx, val_idx) tuples
    """
    if stratify_col is not None and stratify_col in df.columns:
        try:
            kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(kf.split(df, df[stratify_col], groups=df[group_col]))
        except Exception:
            kf = GroupKFold(n_splits=n_splits)
            splits = list(kf.split(df, groups=df[group_col]))
    else:
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(df, groups=df[group_col]))

    return splits


def create_leak_free_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    patient_col: str = 'patient_id',
    spec_col: str = 'spectrogram_id'
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    创建无泄露的数据划分

    确保同一患者和同一Spectrogram不同时出现在训练和验证集

    Args:
        df: DataFrame
        n_splits: 折数
        patient_col: 患者ID列
        spec_col: Spectrogram ID列

    Returns:
        List of (train_idx, val_idx) tuples
    """
    # 首先按患者分组
    gkf = GroupKFold(n_splits=n_splits)

    folds = []
    for train_idx, val_idx in gkf.split(df, groups=df[patient_col]):
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

        # 检查Spectrogram泄露
        train_specs = set(df.iloc[train_idx][spec_col])
        val_specs = set(df.iloc[val_idx][spec_col])
        leak_specs = train_specs & val_specs

        if len(leak_specs) > 0:
            # 从训练集中移除泄露的样本
            leak_mask = df.iloc[train_idx][spec_col].isin(leak_specs)
            train_idx = train_idx[~leak_mask.values]

        folds.append((train_idx, val_idx))

    return folds


def get_high_confidence_samples(
    df: pd.DataFrame,
    threshold: float = 0.7,
    vote_cols: List[str] = None
) -> pd.DataFrame:
    """
    获取高置信度样本

    Args:
        df: DataFrame
        threshold: 置信度阈值
        vote_cols: 投票列名

    Returns:
        高置信度样本DataFrame
    """
    if vote_cols is None:
        vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote',
                     'lrda_vote', 'grda_vote', 'other_vote']

    votes = df[vote_cols].values
    total = votes.sum(axis=1, keepdims=True)
    max_ratio = votes.max(axis=1) / (total.flatten() + 1e-8)

    return df[max_ratio > threshold].reset_index(drop=True)


def compute_class_weights(
    df: pd.DataFrame,
    vote_cols: List[str] = None
) -> np.ndarray:
    """
    计算类别权重（用于处理类别不平衡）

    Args:
        df: DataFrame
        vote_cols: 投票列名

    Returns:
        类别权重数组
    """
    if vote_cols is None:
        vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote',
                     'lrda_vote', 'grda_vote', 'other_vote']

    # 计算每个类别的总投票数
    total_votes = df[vote_cols].sum().values
    total = total_votes.sum()

    # 逆频率权重
    weights = total / (len(vote_cols) * total_votes + 1e-8)

    return weights.astype(np.float32)


def create_soft_labels(row: pd.Series, vote_cols: List[str] = None) -> np.ndarray:
    """
    创建软标签（概率分布）

    Args:
        row: DataFrame行
        vote_cols: 投票列名

    Returns:
        归一化的概率分布
    """
    if vote_cols is None:
        vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote',
                     'lrda_vote', 'grda_vote', 'other_vote']

    votes = row[vote_cols].values.astype(np.float32)
    total = votes.sum()

    if total > 0:
        return votes / total
    else:
        return np.ones(len(vote_cols), dtype=np.float32) / len(vote_cols)


def get_device() -> str:
    """获取可用设备"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数量

    Args:
        model: PyTorch模型

    Returns:
        参数统计字典
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def print_model_summary(model: torch.nn.Module, input_shapes: Dict[str, Tuple] = None):
    """
    打印模型摘要

    Args:
        model: PyTorch模型
        input_shapes: 输入形状字典
    """
    params = count_parameters(model)

    print("="*60)
    print("Model Summary")
    print("="*60)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters: {params['frozen']:,}")
    print("="*60)

    if input_shapes:
        print("\nInput shapes:")
        for name, shape in input_shapes.items():
            print(f"  {name}: {shape}")


class AverageMeter:
    """平均值计算器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停器"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
