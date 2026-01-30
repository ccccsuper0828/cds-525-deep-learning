"""
EEG编码器模块
包含EEGMamba和备用WaveNet实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 尝试导入Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class WaveNetBlock(nn.Module):
    """WaveNet残差块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + residual)
        return x


class WaveNetEncoder(nn.Module):
    """
    1D WaveNet编码器

    用于处理原始EEG信号
    """

    def __init__(
        self,
        in_channels: int = 20,
        hidden_dim: int = 128,
        out_dim: int = 256,
        n_layers: int = 6
    ):
        super().__init__()

        # 初始投影
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)

        # WaveNet blocks with increasing dilation
        dilations = [2 ** i for i in range(n_layers)]
        self.blocks = nn.ModuleList([
            WaveNetBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=d)
            for d in dilations
        ])

        # 输出投影
        self.output_conv = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, T) EEG信号

        Returns:
            (B, out_dim) 特征向量
        """
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.norm(x)

        return x


class BidirectionalMamba(nn.Module):
    """双向Mamba模块"""

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()

        self.d_model = d_model

        if MAMBA_AVAILABLE:
            self.mamba_fwd = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.mamba_bwd = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.use_mamba = True
        else:
            # Fallback to BiLSTM
            self.lstm = nn.LSTM(
                d_model, d_model // 2,
                bidirectional=True,
                batch_first=True,
                num_layers=1
            )
            self.use_mamba = False

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)

        Returns:
            (B, T, D)
        """
        residual = x

        if self.use_mamba:
            fwd = self.mamba_fwd(x)
            bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
            x = fwd + bwd
        else:
            x, _ = self.lstm(x)

        x = self.dropout(x)
        x = self.norm(x + residual)

        return x


class MixtureOfExperts(nn.Module):
    """Mixture of Experts门控"""

    def __init__(self, d_model: int = 256, n_experts: int = 4):
        super().__init__()

        self.n_experts = n_experts

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, n_experts),
            nn.Softmax(dim=-1)
        )

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, D) or (B, T, D)

        Returns:
            Same shape as input
        """
        # 如果是序列，先平均
        if x.dim() == 3:
            x_gate = x.mean(dim=1)
        else:
            x_gate = x

        # 计算门控权重
        gate_weights = self.gate(x_gate)  # (B, n_experts)

        # 计算专家输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)

        if x.dim() == 3:
            # (B, T, D, n_experts) * (B, 1, 1, n_experts) -> (B, T, D)
            gate_weights = gate_weights.unsqueeze(1).unsqueeze(2)
            out = (expert_outputs * gate_weights).sum(dim=-1)
        else:
            # (B, D, n_experts) * (B, 1, n_experts) -> (B, D)
            gate_weights = gate_weights.unsqueeze(1)
            out = (expert_outputs * gate_weights).sum(dim=-1)

        return out


class EEGMambaEncoder(nn.Module):
    """
    EEGMamba编码器

    基于2025 SOTA:
    - EEGMamba (ICLR 2025)
    - FEMBA (arXiv 2025)

    Features:
    - 时空自适应模块
    - 双向Mamba (或BiLSTM fallback)
    - Mixture of Experts
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        n_layers: int = 4,
        out_dim: int = 256,
        use_moe: bool = True,
        n_experts: int = 4
    ):
        super().__init__()

        self.d_model = d_model

        # 时空自适应模块
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=15, padding=7, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 2000, d_model) * 0.02)

        # Bidirectional Mamba layers
        self.mamba_layers = nn.ModuleList([
            BidirectionalMamba(d_model) for _ in range(n_layers)
        ])

        # MoE
        self.use_moe = use_moe
        if use_moe:
            self.moe = MixtureOfExperts(d_model, n_experts)

        # 输出
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: (B, C, T) EEG信号, C=20 channels, T=2000 samples

        Returns:
            (B, out_dim) 特征向量
        """
        B, C, T = x.shape

        # 空间卷积
        x = self.spatial_conv(x)  # (B, D, T)

        # 时间卷积
        x = self.temporal_conv(x)  # (B, D, T)

        # 转换为序列格式
        x = x.permute(0, 2, 1)  # (B, T, D)

        # 添加位置编码
        if T <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :T, :]
        else:
            # 如果序列太长，插值位置编码
            pos = F.interpolate(
                self.pos_encoding.permute(0, 2, 1),
                size=T,
                mode='linear'
            ).permute(0, 2, 1)
            x = x + pos

        # Bidirectional Mamba layers
        for mamba in self.mamba_layers:
            x = mamba(x)

        # 全局平均池化
        x = x.mean(dim=1)  # (B, D)

        # MoE
        if self.use_moe:
            x = self.moe(x)

        # 输出投影
        x = self.dropout(x)
        x = self.fc(x)
        x = self.norm(x)

        return x


def create_eeg_encoder(
    encoder_type: str = 'mamba',
    in_channels: int = 20,
    out_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    创建EEG编码器

    Args:
        encoder_type: 'mamba' or 'wavenet'
        in_channels: 输入通道数
        out_dim: 输出维度
        **kwargs: 其他参数

    Returns:
        EEG编码器模块
    """
    if encoder_type == 'mamba':
        return EEGMambaEncoder(
            in_channels=in_channels,
            out_dim=out_dim,
            **kwargs
        )
    elif encoder_type == 'wavenet':
        return WaveNetEncoder(
            in_channels=in_channels,
            out_dim=out_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
