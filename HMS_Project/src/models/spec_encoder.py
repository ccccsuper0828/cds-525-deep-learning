"""
Spectrogram编码器模块
包含CMFViT (CNN + ViT融合) 和EfficientNet实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed, using basic CNN encoder")


class BasicCNNEncoder(nn.Module):
    """基础CNN编码器（当timm不可用时使用）"""

    def __init__(self, in_channels: int = 4, out_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(256, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.norm(x)
        return x


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet编码器

    用于处理Spectrogram图像
    """

    def __init__(
        self,
        model_name: str = 'tf_efficientnetv2_s',
        in_channels: int = 4,
        out_dim: int = 256,
        pretrained: bool = True,
        drop_rate: float = 0.2
    ):
        super().__init__()

        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=0,
                drop_rate=drop_rate
            )
            backbone_dim = self.backbone.num_features
        else:
            self.backbone = BasicCNNEncoder(in_channels, 512)
            backbone_dim = 512

        self.fc = nn.Linear(backbone_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) Spectrogram图像

        Returns:
            (B, out_dim) 特征向量
        """
        if TIMM_AVAILABLE:
            x = self.backbone(x)
        else:
            x = self.backbone(x)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.norm(x)

        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer编码器

    用于捕捉全局特征
    """

    def __init__(
        self,
        model_name: str = 'vit_small_patch16_224',
        in_channels: int = 4,
        out_dim: int = 256,
        pretrained: bool = True,
        drop_rate: float = 0.1
    ):
        super().__init__()

        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=0,
                drop_rate=drop_rate
            )
            backbone_dim = self.backbone.num_features
        else:
            # Fallback to CNN
            self.backbone = BasicCNNEncoder(in_channels, 384)
            backbone_dim = 384

        self.fc = nn.Linear(backbone_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - will be resized to 224x224

        Returns:
            (B, out_dim)
        """
        # Resize to 224x224 for ViT
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        if TIMM_AVAILABLE:
            x = self.backbone(x)
        else:
            x = self.backbone(x)

        x = self.fc(x)
        x = self.norm(x)

        return x


class CMFViTEncoder(nn.Module):
    """
    CNN-Mamba-ViT融合编码器

    基于2025 SOTA: CMFViT (J Transl Med 2025)

    Features:
    - CNN分支：提取局部特征
    - ViT分支：提取全局特征
    - 多流特征融合
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_dim: int = 256,
        cnn_model: str = 'tf_efficientnetv2_s',
        vit_model: str = 'vit_small_patch16_224',
        pretrained: bool = True,
        fusion_type: str = 'concat'  # 'concat', 'add', 'attention'
    ):
        super().__init__()

        self.fusion_type = fusion_type

        # CNN分支
        self.cnn_encoder = EfficientNetEncoder(
            model_name=cnn_model,
            in_channels=in_channels,
            out_dim=out_dim,
            pretrained=pretrained
        )

        # ViT分支
        self.vit_encoder = ViTEncoder(
            model_name=vit_model,
            in_channels=in_channels,
            out_dim=out_dim,
            pretrained=pretrained
        )

        # 融合层
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim * 2),
                nn.LayerNorm(out_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(out_dim * 2, out_dim)
            )
        elif fusion_type == 'attention':
            self.fusion_attn = nn.MultiheadAttention(out_dim, num_heads=4, batch_first=True)
            self.fusion_norm = nn.LayerNorm(out_dim)
            self.fusion_fc = nn.Linear(out_dim, out_dim)
        else:
            self.fusion = nn.Identity()

        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) Spectrogram图像

        Returns:
            (B, out_dim) 融合特征
        """
        # CNN特征
        cnn_feat = self.cnn_encoder(x)  # (B, out_dim)

        # ViT特征
        vit_feat = self.vit_encoder(x)  # (B, out_dim)

        # 融合
        if self.fusion_type == 'concat':
            combined = torch.cat([cnn_feat, vit_feat], dim=-1)
            out = self.fusion(combined)
        elif self.fusion_type == 'add':
            out = cnn_feat + vit_feat
        elif self.fusion_type == 'attention':
            # 使用交叉注意力融合
            query = cnn_feat.unsqueeze(1)  # (B, 1, D)
            kv = torch.stack([cnn_feat, vit_feat], dim=1)  # (B, 2, D)
            attn_out, _ = self.fusion_attn(query, kv, kv)
            out = self.fusion_norm(attn_out.squeeze(1) + cnn_feat)
            out = self.fusion_fc(out)
        else:
            out = cnn_feat + vit_feat

        out = self.out_norm(out)

        return out


class MultiScaleSpecEncoder(nn.Module):
    """
    多尺度Spectrogram编码器

    处理不同分辨率的输入
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_dim: int = 256,
        scales: Tuple[int, ...] = (64, 128, 256)
    ):
        super().__init__()

        self.scales = scales

        # 每个尺度的编码器
        self.encoders = nn.ModuleList([
            EfficientNetEncoder(
                model_name='efficientnet_b0',
                in_channels=in_channels,
                out_dim=out_dim // len(scales),
                pretrained=True
            ) for _ in scales
        ])

        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, out_dim)
        """
        features = []

        for scale, encoder in zip(self.scales, self.encoders):
            x_scaled = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
            feat = encoder(x_scaled)
            features.append(feat)

        combined = torch.cat(features, dim=-1)
        out = self.fusion(combined)

        return out


def create_spec_encoder(
    encoder_type: str = 'cmfvit',
    in_channels: int = 4,
    out_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    创建Spectrogram编码器

    Args:
        encoder_type: 'cmfvit', 'efficientnet', 'vit', 'multiscale'
        in_channels: 输入通道数
        out_dim: 输出维度

    Returns:
        Spectrogram编码器模块
    """
    if encoder_type == 'cmfvit':
        return CMFViTEncoder(
            in_channels=in_channels,
            out_dim=out_dim,
            **kwargs
        )
    elif encoder_type == 'efficientnet':
        return EfficientNetEncoder(
            in_channels=in_channels,
            out_dim=out_dim,
            **kwargs
        )
    elif encoder_type == 'vit':
        return ViTEncoder(
            in_channels=in_channels,
            out_dim=out_dim,
            **kwargs
        )
    elif encoder_type == 'multiscale':
        return MultiScaleSpecEncoder(
            in_channels=in_channels,
            out_dim=out_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
