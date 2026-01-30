"""
HMS损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class KLDivergenceLoss(nn.Module):
    """KL散度损失"""

    def __init__(self, reduction: str = 'batchmean', eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C) 预测概率
            target: (B, C) 目标概率分布

        Returns:
            KL散度损失
        """
        pred = torch.clamp(pred, min=self.eps, max=1.0 - self.eps)
        target = torch.clamp(target, min=self.eps, max=1.0 - self.eps)

        return F.kl_div(pred.log(), target, reduction=self.reduction)


class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比学习损失

    让相同类别的样本在特征空间中靠近
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (B, D) 归一化后的特征
            labels: (B, C) 软标签（概率分布）

        Returns:
            对比学习损失
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # 使用argmax作为伪类别
        pseudo_labels = labels.argmax(dim=-1)

        # 计算相似度矩阵
        features = F.normalize(features, dim=-1)
        sim_matrix = torch.mm(features, features.t()) / self.temperature

        # 创建正样本mask（同类别）
        labels_eq = pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)
        # 排除自身
        mask = labels_eq.float() - torch.eye(batch_size, device=device)
        mask = torch.clamp(mask, min=0)

        # 计算损失
        # log_prob = sim - log(sum(exp(sim)))
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=-1, keepdim=True) + self.eps)

        # 对每个样本，计算与同类样本的平均log概率
        n_positives = mask.sum(dim=-1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=-1) / (n_positives + self.eps)

        # 只对有正样本的计算损失
        valid_mask = n_positives > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -mean_log_prob_pos[valid_mask].mean()

        return loss


class CrossModalContrastiveLoss(nn.Module):
    """
    跨模态对比学习损失 (CLIP-style)

    2025 SOTA: 让同一样本的不同模态特征在特征空间中对齐
    参考: CLIP (OpenAI), ImageBind (Meta 2023), UniModal (2024)

    核心思想：
    - 同一样本的EEG和Spectrogram特征应该相似
    - 不同样本的特征应该不相似
    - 双向对比：EEG→Spec 和 Spec→EEG
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        eeg_features: torch.Tensor,
        spec_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            eeg_features: (B, D) EEG编码特征
            spec_features: (B, D) Spectrogram编码特征

        Returns:
            跨模态对比损失
        """
        device = eeg_features.device
        batch_size = eeg_features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # 归一化
        eeg_features = F.normalize(eeg_features, dim=-1)
        spec_features = F.normalize(spec_features, dim=-1)

        # 计算相似度矩阵 (B, B)
        # logits[i,j] = eeg_i · spec_j / τ
        logits = torch.mm(eeg_features, spec_features.t()) / self.temperature

        # 标签：对角线为正样本 (同一样本的不同模态)
        labels = torch.arange(batch_size, device=device)

        # 双向对比损失
        # EEG → Spec: 给定EEG，找对应的Spec
        loss_eeg_to_spec = F.cross_entropy(logits, labels)
        # Spec → EEG: 给定Spec，找对应的EEG
        loss_spec_to_eeg = F.cross_entropy(logits.t(), labels)

        # 平均
        loss = (loss_eeg_to_spec + loss_spec_to_eeg) / 2

        return loss


class SoftLabelContrastiveLoss(nn.Module):
    """
    软标签对比学习损失

    2025 SOTA改进: 使用软标签计算样本间相似度，而不是argmax伪标签
    参考: Soft Contrastive Learning (NeurIPS 2023)

    优势：
    - 保留标签的不确定性信息
    - 相似软标签的样本被拉近（不仅仅是相同类别）
    - 更适合HMS这种多专家投票的软标签场景
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        soft_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (B, D) 特征
            soft_labels: (B, C) 软标签（概率分布）

        Returns:
            软标签对比损失
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # 归一化特征
        features = F.normalize(features, dim=-1)

        # 计算特征相似度矩阵 (B, B)
        feature_sim = torch.mm(features, features.t()) / self.temperature

        # 计算软标签相似度矩阵 (B, B)
        # 使用余弦相似度衡量标签分布的相似性
        soft_labels_norm = F.normalize(soft_labels, dim=-1)
        label_sim = torch.mm(soft_labels_norm, soft_labels_norm.t())

        # 将标签相似度转换为软目标（排除自身）
        mask = 1.0 - torch.eye(batch_size, device=device)
        label_sim = label_sim * mask

        # 归一化为概率分布
        soft_targets = F.softmax(label_sim / self.temperature, dim=-1)

        # 计算特征相似度的log_softmax（排除自身）
        feature_sim = feature_sim * mask + torch.eye(batch_size, device=device) * (-1e9)
        log_probs = F.log_softmax(feature_sim, dim=-1)

        # KL散度损失：让特征相似度分布匹配标签相似度分布
        loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')

        return loss


class MultiModalContrastiveLoss(nn.Module):
    """
    多模态综合对比学习损失

    整合三种对比学习策略：
    1. 跨模态对比 (Cross-Modal): EEG ↔ Spec 对齐
    2. 软标签对比 (Soft-Label): 利用投票不确定性
    3. 监督对比 (Supervised): 同类别聚集

    2025 SOTA组合策略
    """

    def __init__(
        self,
        temperature: float = 0.07,
        cross_modal_weight: float = 0.3,
        soft_label_weight: float = 0.3,
        supervised_weight: float = 0.4
    ):
        super().__init__()
        self.cross_modal_loss = CrossModalContrastiveLoss(temperature)
        self.soft_label_loss = SoftLabelContrastiveLoss(temperature)
        self.supervised_loss = SupervisedContrastiveLoss(temperature)

        self.cross_modal_weight = cross_modal_weight
        self.soft_label_weight = soft_label_weight
        self.supervised_weight = supervised_weight

    def forward(
        self,
        fused_features: torch.Tensor,
        eeg_features: torch.Tensor,
        spec_features: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: (B, D) 融合后特征
            eeg_features: (B, D) EEG编码特征
            spec_features: (B, D) Spectrogram编码特征
            labels: (B, C) 软标签

        Returns:
            Dict containing individual and total losses
        """
        losses = {}

        # 1. 跨模态对比损失
        cross_modal = self.cross_modal_loss(eeg_features, spec_features)
        losses['cross_modal'] = cross_modal

        # 2. 软标签对比损失
        soft_label = self.soft_label_loss(fused_features, labels)
        losses['soft_label'] = soft_label

        # 3. 监督对比损失
        supervised = self.supervised_loss(fused_features, labels)
        losses['supervised'] = supervised

        # 加权总和
        total = (
            self.cross_modal_weight * cross_modal +
            self.soft_label_weight * soft_label +
            self.supervised_weight * supervised
        )
        losses['total'] = total

        return losses


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""

    def __init__(self, smoothing: float = 0.1, num_classes: int = 6):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C) 预测logits或概率
            target: (B, C) 目标概率分布

        Returns:
            平滑后的交叉熵损失
        """
        # 应用标签平滑
        smooth_target = (1 - self.smoothing) * target + self.smoothing / self.num_classes

        # 计算交叉熵
        log_pred = F.log_softmax(pred, dim=-1)
        loss = -(smooth_target * log_pred).sum(dim=-1).mean()

        return loss


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C) 预测概率
            target: (B, C) 目标概率分布

        Returns:
            Focal loss
        """
        # 计算交叉熵
        ce_loss = -(target * torch.log(pred + 1e-8)).sum(dim=-1)

        # 计算focal权重
        pt = (pred * target).sum(dim=-1)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class HMSLoss(nn.Module):
    """
    HMS综合损失函数 (2025 SOTA版本)

    组成:
    1. KL散度损失（主任务）
    2. 跨模态对比学习损失（EEG↔Spec对齐）【新增】
    3. 软标签对比学习损失（利用投票不确定性）【新增】
    4. 监督对比学习损失（同类聚集）
    5. 抗模态坍塌损失
    6. 标签平滑（可选）
    """

    def __init__(
        self,
        use_kl: bool = True,
        use_contrastive: bool = True,
        use_cross_modal: bool = True,
        use_soft_label_contrastive: bool = True,
        use_collapse: bool = True,
        use_label_smoothing: bool = False,
        kl_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        cross_modal_weight: float = 0.05,
        soft_label_weight: float = 0.05,
        collapse_weight: float = 0.01,
        smoothing: float = 0.1,
        temperature: float = 0.07,
        num_classes: int = 6
    ):
        super().__init__()

        self.use_kl = use_kl
        self.use_contrastive = use_contrastive
        self.use_cross_modal = use_cross_modal
        self.use_soft_label_contrastive = use_soft_label_contrastive
        self.use_collapse = use_collapse
        self.use_label_smoothing = use_label_smoothing

        self.kl_weight = kl_weight
        self.contrastive_weight = contrastive_weight
        self.cross_modal_weight = cross_modal_weight
        self.soft_label_weight = soft_label_weight
        self.collapse_weight = collapse_weight

        # 损失函数
        if use_kl:
            self.kl_loss = KLDivergenceLoss()

        if use_contrastive:
            self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)

        if use_cross_modal:
            self.cross_modal_loss = CrossModalContrastiveLoss(temperature=temperature)

        if use_soft_label_contrastive:
            self.soft_label_loss = SoftLabelContrastiveLoss(temperature=temperature)

        if use_label_smoothing:
            self.smoothing_loss = LabelSmoothingLoss(smoothing=smoothing, num_classes=num_classes)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        contrastive_features: Optional[torch.Tensor] = None,
        eeg_features: Optional[torch.Tensor] = None,
        spec_features: Optional[torch.Tensor] = None,
        collapse_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            pred: (B, C) 预测概率
            target: (B, C) 目标概率分布
            contrastive_features: (B, D) 融合后对比学习特征（可选）
            eeg_features: (B, D) EEG编码特征（用于跨模态对比）
            spec_features: (B, D) Spec编码特征（用于跨模态对比）
            collapse_loss: 模态坍塌损失（可选）

        Returns:
            Dict containing all losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=pred.device)

        # 1. KL散度损失（主任务）
        if self.use_kl:
            kl = self.kl_loss(pred, target)
            losses['kl'] = kl
            total_loss = total_loss + self.kl_weight * kl
        elif self.use_label_smoothing:
            smooth = self.smoothing_loss(pred, target)
            losses['smoothing'] = smooth
            total_loss = total_loss + smooth

        # 2. 监督对比学习损失
        if self.use_contrastive and contrastive_features is not None:
            contrastive = self.contrastive_loss(contrastive_features, target)
            losses['contrastive'] = contrastive
            total_loss = total_loss + self.contrastive_weight * contrastive

        # 3. 跨模态对比学习损失【新增2025 SOTA】
        if self.use_cross_modal and eeg_features is not None and spec_features is not None:
            cross_modal = self.cross_modal_loss(eeg_features, spec_features)
            losses['cross_modal'] = cross_modal
            total_loss = total_loss + self.cross_modal_weight * cross_modal

        # 4. 软标签对比学习损失【新增2025 SOTA】
        if self.use_soft_label_contrastive and contrastive_features is not None:
            soft_label = self.soft_label_loss(contrastive_features, target)
            losses['soft_label'] = soft_label
            total_loss = total_loss + self.soft_label_weight * soft_label

        # 5. 模态坍塌损失
        if self.use_collapse and collapse_loss is not None:
            losses['collapse'] = collapse_loss
            total_loss = total_loss + self.collapse_weight * collapse_loss

        losses['total'] = total_loss

        return losses


def create_loss_function(
    loss_type: str = 'hms',
    **kwargs
) -> nn.Module:
    """
    创建损失函数

    Args:
        loss_type: 'hms', 'kl', 'focal'

    Returns:
        损失函数模块
    """
    if loss_type == 'hms':
        return HMSLoss(**kwargs)
    elif loss_type == 'kl':
        return KLDivergenceLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
