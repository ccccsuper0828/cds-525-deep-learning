"""
多模态融合模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ModalityQualityEstimator(nn.Module):
    """
    模态质量评估器

    评估每个模态的信息量和可靠性
    参考: AMC (ICML 2025)
    """

    def __init__(self, dim: int = 256, hidden_dim: int = 64):
        super().__init__()

        self.estimator = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, D) 模态特征

        Returns:
            (B, 1) 质量分数 [0, 1]
        """
        return self.estimator(x)


class GatedFusion(nn.Module):
    """
    门控融合模块

    动态计算各模态的贡献权重
    """

    def __init__(self, dim: int = 256, n_modalities: int = 3):
        super().__init__()

        self.n_modalities = n_modalities

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(dim * n_modalities, dim),
            nn.ReLU(),
            nn.Linear(dim, n_modalities),
            nn.Softmax(dim=-1)
        )

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List of (B, D) tensors

        Returns:
            fused: (B, D) 融合特征
            weights: (B, n_modalities) 门控权重
        """
        # 拼接所有特征
        concat = torch.cat(features, dim=-1)  # (B, D * n_modalities)

        # 计算门控权重
        weights = self.gate(concat)  # (B, n_modalities)

        # 加权求和
        fused = sum(w.unsqueeze(-1) * f for w, f in zip(weights.unbind(dim=-1), features))

        return fused, weights


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块

    让不同模态相互关注
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, D) 查询模态
            key_value: (B, N, D) 键值模态（N个token）

        Returns:
            (B, D) 增强后的查询特征
        """
        # 扩展query为序列
        query = query.unsqueeze(1)  # (B, 1, D)

        # 交叉注意力
        attn_out, _ = self.attn(query, key_value, key_value)
        query = self.norm1(query + attn_out)

        # FFN
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query.squeeze(1)


class AdaptiveGatedFusion(nn.Module):
    """
    自适应门控融合模块

    Features:
    - 模态质量评估
    - 动态门控
    - 抗模态坍塌机制

    参考:
    - AMC (ICML 2025)
    - Hate-UDF (2025)
    - Modality Collapse (ICML 2025 Spotlight)
    """

    def __init__(
        self,
        dim: int = 256,
        n_modalities: int = 3,
        use_quality_estimation: bool = True,
        use_cross_attention: bool = True,
        collapse_threshold: float = 0.3
    ):
        super().__init__()

        self.dim = dim
        self.n_modalities = n_modalities
        self.use_quality_estimation = use_quality_estimation
        self.use_cross_attention = use_cross_attention
        self.collapse_threshold = collapse_threshold

        # 模态质量评估器
        if use_quality_estimation:
            self.quality_estimators = nn.ModuleList([
                ModalityQualityEstimator(dim) for _ in range(n_modalities)
            ])

        # 门控网络
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * n_modalities, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            ) for _ in range(n_modalities)
        ])

        # 跨模态注意力
        if use_cross_attention:
            self.cross_attentions = nn.ModuleList([
                CrossModalAttention(dim) for _ in range(n_modalities)
            ])

        # 融合投影
        self.projection = nn.Sequential(
            nn.Linear(dim * n_modalities, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

        self.output_norm = nn.LayerNorm(dim)

        # 抗模态坍塌监控
        self.register_buffer('modality_ranks', torch.ones(n_modalities))
        self.register_buffer('update_count', torch.tensor(0))

    def forward(
        self,
        features: List[torch.Tensor],
        return_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: List of (B, D) tensors, one per modality
            return_weights: 是否返回融合权重

        Returns:
            fused: (B, D) 融合特征
            weights: (B, n_modalities) 权重（如果return_weights=True）
        """
        batch_size = features[0].shape[0]

        # 1. 计算模态质量权重
        if self.use_quality_estimation:
            quality_weights = []
            for feat, estimator in zip(features, self.quality_estimators):
                q = estimator(feat)  # (B, 1)
                quality_weights.append(q)
            quality_weights = torch.cat(quality_weights, dim=-1)  # (B, n_modalities)
        else:
            quality_weights = torch.ones(batch_size, self.n_modalities, device=features[0].device)

        # 2. 计算门控权重
        concat_feat = torch.cat(features, dim=-1)  # (B, D * n_modalities)
        gate_weights = []
        for gate in self.gates:
            g = gate(concat_feat)  # (B, 1)
            gate_weights.append(g)
        gate_weights = torch.cat(gate_weights, dim=-1)  # (B, n_modalities)

        # 3. 组合质量和门控权重
        combined_weights = quality_weights * gate_weights
        combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. 跨模态注意力增强
        if self.use_cross_attention:
            enhanced_features = []
            for i, (feat, cross_attn) in enumerate(zip(features, self.cross_attentions)):
                # 其他模态作为键值
                other_feats = torch.stack([f for j, f in enumerate(features) if j != i], dim=1)
                enhanced = cross_attn(feat, other_feats)
                enhanced_features.append(enhanced)
            features = enhanced_features

        # 5. 加权融合
        weighted_features = []
        for i, feat in enumerate(features):
            w = combined_weights[:, i:i+1]  # (B, 1)
            weighted_features.append(feat * w)

        # 拼接并投影
        fused = torch.cat(weighted_features, dim=-1)
        fused = self.projection(fused)
        fused = self.output_norm(fused)

        # 6. 更新模态有效秩（用于监控坍塌）
        if self.training:
            self._update_modality_ranks(features)

        if return_weights:
            return fused, combined_weights
        return fused, None

    def _update_modality_ranks(self, features: List[torch.Tensor]):
        """更新模态有效秩统计"""
        with torch.no_grad():
            for i, feat in enumerate(features):
                rank = self._compute_effective_rank(feat)
                # 指数移动平均
                self.modality_ranks[i] = 0.99 * self.modality_ranks[i] + 0.01 * rank
            self.update_count += 1

    def _compute_effective_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算特征矩阵的有效秩

        有效秩越低，表示特征越"坍塌"到低维空间
        """
        try:
            # SVD分解
            _, s, _ = torch.svd(x)
            # 归一化奇异值
            s = s / (s.sum() + 1e-8)
            # 计算熵
            entropy = -(s * torch.log(s + 1e-8)).sum()
            # 有效秩 = exp(entropy)
            return torch.exp(entropy)
        except Exception:
            return torch.tensor(self.dim, device=x.device, dtype=x.dtype)

    def get_collapse_loss(self) -> torch.Tensor:
        """
        计算抗模态坍塌损失

        惩罚模态间有效秩差异过大
        """
        if self.update_count < 100:
            return torch.tensor(0.0, device=self.modality_ranks.device)

        # 计算有效秩的标准差
        rank_std = self.modality_ranks.std()

        # 检查是否有模态坍塌
        min_rank = self.modality_ranks.min()
        max_rank = self.modality_ranks.max()

        # 如果最小秩太小，额外惩罚
        collapse_penalty = F.relu(self.collapse_threshold * self.dim - min_rank)

        return rank_std + collapse_penalty

    def get_modality_stats(self) -> dict:
        """获取模态统计信息"""
        return {
            'ranks': self.modality_ranks.detach().cpu().numpy(),
            'update_count': self.update_count.item()
        }


class SimpleConcatFusion(nn.Module):
    """简单拼接融合（基线）"""

    def __init__(self, dim: int = 256, n_modalities: int = 3):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(dim * n_modalities, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, features: List[torch.Tensor], return_weights: bool = False):
        concat = torch.cat(features, dim=-1)
        fused = self.projection(concat)

        if return_weights:
            weights = torch.ones(fused.shape[0], len(features), device=fused.device) / len(features)
            return fused, weights
        return fused, None


def create_fusion_module(
    fusion_type: str = 'adaptive',
    dim: int = 256,
    n_modalities: int = 3,
    **kwargs
) -> nn.Module:
    """
    创建融合模块

    Args:
        fusion_type: 'adaptive', 'gated', 'concat'
        dim: 特征维度
        n_modalities: 模态数量

    Returns:
        融合模块
    """
    if fusion_type == 'adaptive':
        return AdaptiveGatedFusion(dim, n_modalities, **kwargs)
    elif fusion_type == 'gated':
        return GatedFusion(dim, n_modalities)
    elif fusion_type == 'concat':
        return SimpleConcatFusion(dim, n_modalities)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
