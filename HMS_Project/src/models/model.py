"""
HMS完整模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .eeg_encoder import EEGMambaEncoder, WaveNetEncoder, create_eeg_encoder
from .spec_encoder import CMFViTEncoder, EfficientNetEncoder, create_spec_encoder
from .fusion import AdaptiveGatedFusion, create_fusion_module


class ContrastiveHead(nn.Module):
    """对比学习投影头"""

    def __init__(self, in_dim: int = 256, out_dim: int = 128):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class ClassificationHead(nn.Module):
    """分类头"""

    def __init__(
        self,
        in_dim: int = 256,
        num_classes: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class HMSModel(nn.Module):
    """
    HMS有害脑活动分类模型

    架构:
    - EEG编码器: EEGMamba或WaveNet
    - EEG频谱图编码器: CMFViT
    - 原始频谱图编码器: CMFViT
    - 自适应门控融合
    - 对比学习头（可选）
    - 分类头

    基于2025 SOTA:
    - EEGMamba (ICLR 2025)
    - FEMBA (arXiv 2025)
    - CMFViT (J Transl Med 2025)
    - AMC Fusion (ICML 2025)
    """

    def __init__(
        self,
        num_classes: int = 6,
        dim: int = 256,
        eeg_encoder_type: str = 'mamba',
        spec_encoder_type: str = 'cmfvit',
        fusion_type: str = 'adaptive',
        use_eeg_spec: bool = True,
        use_contrastive: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            num_classes: 分类类别数
            dim: 特征维度
            eeg_encoder_type: EEG编码器类型 ('mamba', 'wavenet')
            spec_encoder_type: Spectrogram编码器类型 ('cmfvit', 'efficientnet')
            fusion_type: 融合类型 ('adaptive', 'gated', 'concat')
            use_eeg_spec: 是否使用EEG生成的频谱图
            use_contrastive: 是否使用对比学习
            dropout: Dropout率
        """
        super().__init__()

        self.use_eeg_spec = use_eeg_spec
        self.use_contrastive = use_contrastive
        self.dim = dim

        # EEG编码器
        self.eeg_encoder = create_eeg_encoder(
            encoder_type=eeg_encoder_type,
            in_channels=20,
            out_dim=dim
        )

        # EEG频谱图编码器
        if use_eeg_spec:
            self.eeg_spec_encoder = create_spec_encoder(
                encoder_type=spec_encoder_type,
                in_channels=20,  # 20个EEG通道的频谱图
                out_dim=dim
            )
            n_modalities = 3
        else:
            self.eeg_spec_encoder = None
            n_modalities = 2

        # 原始频谱图编码器
        self.spec_encoder = create_spec_encoder(
            encoder_type=spec_encoder_type,
            in_channels=4,  # 4个脑区
            out_dim=dim
        )

        # 融合模块
        self.fusion = create_fusion_module(
            fusion_type=fusion_type,
            dim=dim,
            n_modalities=n_modalities
        )

        # 对比学习头
        if use_contrastive:
            self.contrastive_head = ContrastiveHead(dim, 128)

        # 分类头
        self.classifier = ClassificationHead(
            in_dim=dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(
        self,
        eeg: torch.Tensor,
        spec: torch.Tensor,
        eeg_spec: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            eeg: (B, 20, T) 原始EEG
            spec: (B, 4, H, W) 原始频谱图
            eeg_spec: (B, 20, H, W) EEG生成的频谱图（可选）
            return_features: 是否返回中间特征

        Returns:
            Dict containing:
            - 'probs': (B, num_classes) 预测概率
            - 'logits': (B, num_classes) 原始logits
            - 'features': (B, dim) 融合特征（如果return_features）
            - 'contrastive_features': (B, 128) 对比学习特征（如果use_contrastive）
            - 'eeg_features': (B, dim) EEG编码特征（用于跨模态对比）
            - 'spec_features': (B, dim) Spec编码特征（用于跨模态对比）
            - 'fusion_weights': (B, n_modalities) 融合权重
        """
        # 编码各模态
        eeg_feat = self.eeg_encoder(eeg)  # (B, dim)
        spec_feat = self.spec_encoder(spec)  # (B, dim)

        # 收集所有模态特征
        if self.use_eeg_spec and eeg_spec is not None:
            eeg_spec_feat = self.eeg_spec_encoder(eeg_spec)  # (B, dim)
            features = [eeg_feat, eeg_spec_feat, spec_feat]
        else:
            features = [eeg_feat, spec_feat]

        # 融合
        fused, fusion_weights = self.fusion(features, return_weights=True)

        # 分类
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        # 构建输出
        output = {
            'probs': probs,
            'logits': logits,
            'fusion_weights': fusion_weights
        }

        if return_features:
            output['features'] = fused
            # 返回各模态原始特征（用于跨模态对比学习）
            output['eeg_features'] = eeg_feat
            output['spec_features'] = spec_feat

        if self.use_contrastive:
            contrastive_feat = self.contrastive_head(fused)
            output['contrastive_features'] = contrastive_feat

        return output

    def get_collapse_loss(self) -> torch.Tensor:
        """获取抗模态坍塌损失"""
        if hasattr(self.fusion, 'get_collapse_loss'):
            return self.fusion.get_collapse_loss()
        return torch.tensor(0.0)

    def get_fusion_stats(self) -> dict:
        """获取融合模块统计信息"""
        if hasattr(self.fusion, 'get_modality_stats'):
            return self.fusion.get_modality_stats()
        return {}


class HMSModelLite(nn.Module):
    """
    HMS轻量级模型

    用于快速实验和资源受限环境
    """

    def __init__(
        self,
        num_classes: int = 6,
        dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        # 轻量级EEG编码器
        self.eeg_encoder = WaveNetEncoder(
            in_channels=20,
            hidden_dim=64,
            out_dim=dim,
            n_layers=4
        )

        # 轻量级频谱图编码器
        self.spec_encoder = EfficientNetEncoder(
            model_name='efficientnet_b0',
            in_channels=4,
            out_dim=dim,
            pretrained=True
        )

        # 简单融合
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 分类头
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, eeg, spec, eeg_spec=None, return_features=False):
        eeg_feat = self.eeg_encoder(eeg)
        spec_feat = self.spec_encoder(spec)

        combined = torch.cat([eeg_feat, spec_feat], dim=-1)
        fused = self.fusion(combined)

        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        output = {
            'probs': probs,
            'logits': logits,
            'fusion_weights': None
        }

        if return_features:
            output['features'] = fused

        return output


class HMSEnsemble(nn.Module):
    """
    HMS模型集成

    用于推理时的多模型集成
    """

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()

        self.models = nn.ModuleList(models)
        self.n_models = len(models)

        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models
        self.register_buffer('weights', torch.tensor(weights))

    def forward(self, eeg, spec, eeg_spec=None, return_features=False):
        all_probs = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(eeg, spec, eeg_spec, return_features=False)
                all_probs.append(output['probs'])

        # 加权平均
        probs = torch.stack(all_probs, dim=0)  # (n_models, B, num_classes)
        weights = self.weights.view(-1, 1, 1)
        probs = (probs * weights).sum(dim=0)

        return {
            'probs': probs,
            'logits': torch.log(probs + 1e-8),
            'fusion_weights': None
        }


def create_model(
    model_type: str = 'full',
    num_classes: int = 6,
    **kwargs
) -> nn.Module:
    """
    创建HMS模型

    Args:
        model_type: 'full', 'lite'
        num_classes: 分类类别数
        **kwargs: 其他参数

    Returns:
        HMS模型
    """
    if model_type == 'full':
        return HMSModel(num_classes=num_classes, **kwargs)
    elif model_type == 'lite':
        return HMSModelLite(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
