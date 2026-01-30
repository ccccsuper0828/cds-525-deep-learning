# HMS有害脑活动分类 - 完整实验方案 (2025 SOTA)

## 一、项目概述

### 1.1 任务描述

| 项目 | 内容 |
|------|------|
| **比赛** | HMS - Harmful Brain Activity Classification (Kaggle) |
| **主办方** | Harvard Medical School |
| **任务** | 预测专家对EEG信号中6类有害脑活动的投票概率分布 |
| **评价指标** | KL散度 (KL-Divergence) |
| **第1名成绩** | 0.272332 (Team Sony) |

### 1.2 数据规模

| 项目 | 数量 |
|------|------|
| 总样本数 | 106,800 |
| EEG文件 | 17,089 |
| Spectrogram文件 | 11,138 |
| 患者数 | 1,950 |
| 类别 | 6类 (Seizure, LPD, GPD, LRDA, GRDA, Other) |

### 1.3 数据特点与挑战

```
⚠️ 关键问题：
1. 患者数据量严重不平衡 (1~2215样本/患者)
2. EEG和Spectrogram是独立数据源，offset可能不同
3. 多个EEG可共享同一Spectrogram
4. 投票数不均匀 (1~22票)
```

---

## 二、2025年SOTA技术架构

### 2.1 技术选型对比

| 模型 | 类型 | 参数量 | 特点 | 发表 | 推荐度 |
|------|------|--------|------|------|--------|
| **FEMBA** | Mamba Foundation | 7.8M-389M | 21000+小时预训练，线性复杂度 | arXiv 2025 | ⭐⭐⭐⭐⭐ |
| **EEGMamba** | Mamba+MoE | ~50M | 多任务学习，7个数据集SOTA | ICLR 2025 | ⭐⭐⭐⭐⭐ |
| **LaBraM** | Transformer | 5.8M-369M | 2500小时预训练，神经tokenizer | ICLR 2024 Spotlight | ⭐⭐⭐⭐ |
| **EEGPT** | Transformer | 10M | 双自监督学习 | ICLR 2024 | ⭐⭐⭐⭐ |
| **LCM** | Transformer | - | 超越LaBraM/EEGPT | arXiv 2025 | ⭐⭐⭐⭐ |
| **CMFViT** | ViT+CNN | ~30M | 多流融合，98.85%准确率 | J Transl Med 2025 | ⭐⭐⭐⭐ |
| **EfficientNetV2** | CNN | 20M | Kaggle获奖方案核心 | - | ⭐⭐⭐⭐ |

### 2.2 推荐架构：多模态Mamba-Transformer混合架构

```
                            输入
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         [EEG分支]     [EEG频谱图]    [Spectrogram分支]
         20ch×2000     自生成STFT      4区×128×256
              │              │              │
              ▼              ▼              ▼
       ┌───────────┐  ┌───────────┐  ┌───────────┐
       │ EEGMamba  │  │ CMFViT    │  │EfficientNet│
       │ (Bi-SSM)  │  │ (ViT+CNN) │  │   V2-S    │
       └───────────┘  └───────────┘  └───────────┘
              │              │              │
              ▼              ▼              ▼
          [256-dim]     [256-dim]      [256-dim]
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                   ┌─────────────────┐
                   │ Adaptive Gated  │
                   │ Fusion (AGF)    │
                   │ + MoE Router    │
                   └─────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │ Supervised      │
                   │ Contrastive     │
                   │ Learning        │
                   └─────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │ Classification  │
                   │ Head (6 class)  │
                   └─────────────────┘
                             │
                             ▼
                    KL-Divergence Loss
```

---

## 三、数据处理方案

### 3.1 数据对齐策略（关键）

```python
"""
⚠️ 核心要点：EEG和Spectrogram使用各自的offset独立提取！
"""

class HMSDataProcessor:
    def __init__(self, eeg_dir, spec_dir):
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.eeg_cache = {}
        self.spec_cache = {}

    def extract_eeg(self, eeg_id, eeg_offset, window_sec=10, fs=200):
        """
        从EEG文件中按eeg_label_offset_seconds提取窗口
        """
        if eeg_id not in self.eeg_cache:
            path = f"{self.eeg_dir}/{eeg_id}.parquet"
            self.eeg_cache[eeg_id] = pd.read_parquet(path).values

        eeg_data = self.eeg_cache[eeg_id]

        start = int(eeg_offset * fs)
        end = start + int(window_sec * fs)

        # 边界处理
        if end > len(eeg_data):
            end = len(eeg_data)
            start = max(0, end - int(window_sec * fs))

        window = eeg_data[start:end]

        # Padding
        if len(window) < window_sec * fs:
            pad_len = window_sec * fs - len(window)
            window = np.pad(window, ((0, pad_len), (0, 0)), mode='reflect')

        return window  # (2000, 20)

    def extract_spectrogram(self, spec_id, spec_offset, window_sec=10):
        """
        从Spectrogram文件中按spectrogram_label_offset_seconds提取窗口
        """
        if spec_id not in self.spec_cache:
            path = f"{self.spec_dir}/{spec_id}.parquet"
            self.spec_cache[spec_id] = pd.read_parquet(path)

        spec_df = self.spec_cache[spec_id]
        time_col = spec_df['time'].values

        # 提取offset附近的窗口
        mask = (time_col >= spec_offset) & (time_col < spec_offset + window_sec)

        if mask.sum() == 0:
            # 找最近的帧
            closest = np.abs(time_col - spec_offset).argmin()
            start_idx = max(0, closest - 2)
            end_idx = min(len(time_col), closest + 3)
            mask = np.zeros(len(time_col), dtype=bool)
            mask[start_idx:end_idx] = True

        # 分离4个脑区
        regions = ['LL', 'LP', 'RL', 'RP']
        spec_images = []
        for region in regions:
            cols = [c for c in spec_df.columns if c.startswith(f'{region}_')]
            region_data = spec_df.loc[mask, cols].values
            spec_images.append(region_data)

        spec_window = np.stack(spec_images, axis=0)  # (4, T, F)

        return spec_window

    def generate_eeg_spectrogram(self, eeg_window, fs=200, nperseg=256, noverlap=128):
        """
        从EEG生成自定义频谱图（增强模态）
        """
        from scipy import signal

        spectrograms = []
        for ch in range(eeg_window.shape[1]):
            f, t, Sxx = signal.spectrogram(
                eeg_window[:, ch], fs=fs,
                nperseg=nperseg, noverlap=noverlap
            )
            spectrograms.append(Sxx)

        return np.stack(spectrograms, axis=0)  # (20, F, T)
```

### 3.2 数据增强策略

```python
import numpy as np
from scipy import signal
import albumentations as A

class EEGAugmentation:
    """EEG数据增强"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, eeg):
        # 1. 时间偏移
        if np.random.rand() < self.p:
            shift = np.random.randint(-200, 200)  # ±1秒
            eeg = np.roll(eeg, shift, axis=0)

        # 2. 高斯噪声
        if np.random.rand() < self.p:
            noise = np.random.normal(0, 0.01 * eeg.std(), eeg.shape)
            eeg = eeg + noise

        # 3. 通道Dropout
        if np.random.rand() < self.p:
            n_drop = np.random.randint(1, 4)
            drop_idx = np.random.choice(eeg.shape[1], n_drop, replace=False)
            eeg[:, drop_idx] = 0

        # 4. 时间Mask (SpecAugment风格)
        if np.random.rand() < self.p:
            mask_len = np.random.randint(50, 200)
            mask_start = np.random.randint(0, eeg.shape[0] - mask_len)
            eeg[mask_start:mask_start + mask_len] = 0

        # 5. 带通滤波增强
        if np.random.rand() < self.p * 0.3:
            b, a = signal.butter(4, [0.5, 30], btype='band', fs=200)
            eeg = signal.filtfilt(b, a, eeg, axis=0)

        return eeg


class SpectrogramAugmentation:
    """Spectrogram数据增强"""

    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ])

    def __call__(self, spec):
        # spec: (4, H, W)
        augmented = []
        for i in range(4):
            img = spec[i]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
            aug = self.transform(image=img)['image']
            augmented.append(aug.astype(np.float32) / 255.0)
        return np.stack(augmented, axis=0)


def mixup(eeg1, spec1, label1, eeg2, spec2, label2, alpha=0.4):
    """Mixup数据增强"""
    lam = np.random.beta(alpha, alpha)
    eeg_mix = lam * eeg1 + (1 - lam) * eeg2
    spec_mix = lam * spec1 + (1 - lam) * spec2
    label_mix = lam * label1 + (1 - lam) * label2
    return eeg_mix, spec_mix, label_mix
```

### 3.3 患者平衡采样

```python
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import GroupKFold
import numpy as np

def create_patient_balanced_sampler(df, power=0.5):
    """
    创建患者平衡采样器
    Args:
        df: 训练数据DataFrame
        power: 权重指数，0.5为sqrt平衡
    """
    patient_counts = df.groupby('patient_id').size()
    max_count = patient_counts.max()

    # 权重 = (max_count / patient_count) ^ power
    weights = df['patient_id'].map(
        lambda x: (max_count / patient_counts[x]) ** power
    ).values

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(df),
        replacement=True
    )

    return sampler


def create_leak_free_folds(df, n_splits=5):
    """
    创建无泄露的交叉验证划分
    同时考虑patient_id和spectrogram_id
    """
    # 创建复合分组键
    df = df.copy()
    df['group_key'] = df['patient_id'].astype(str)

    gkf = GroupKFold(n_splits=n_splits)

    folds = []
    for train_idx, val_idx in gkf.split(df, groups=df['patient_id']):
        # 确保验证集的spectrogram不在训练集中
        train_specs = set(df.iloc[train_idx]['spectrogram_id'])
        val_specs = set(df.iloc[val_idx]['spectrogram_id'])

        # 检查泄露
        leak = train_specs & val_specs
        if len(leak) > 0:
            # 从训练集中移除泄露的spectrogram
            leak_mask = df.iloc[train_idx]['spectrogram_id'].isin(leak)
            train_idx = train_idx[~leak_mask.values]

        folds.append((train_idx, val_idx))

    return folds
```

---

## 四、模型架构实现

### 4.1 EEGMamba模块 (2025 SOTA)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed, using fallback LSTM")


class BidirectionalMamba(nn.Module):
    """双向Mamba模块 - 线性复杂度处理长序列"""

    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super().__init__()

        if MAMBA_AVAILABLE:
            self.mamba_fwd = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_bwd = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            # Fallback to BiLSTM
            self.lstm = nn.LSTM(d_model, d_model // 2, bidirectional=True, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.use_mamba = MAMBA_AVAILABLE

    def forward(self, x):
        # x: (B, T, D)
        if self.use_mamba:
            fwd = self.mamba_fwd(x)
            bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
            out = fwd + bwd
        else:
            out, _ = self.lstm(x)

        return self.norm(out + x)


class EEGMambaEncoder(nn.Module):
    """
    EEGMamba编码器 - 基于2025 SOTA
    参考: EEGMamba (ICLR 2025), FEMBA (arXiv 2025)
    """

    def __init__(self, in_channels=20, d_model=256, n_layers=4, out_dim=256):
        super().__init__()

        # 时空自适应模块
        self.spatial_conv = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=15, padding=7, groups=d_model)

        # Bidirectional Mamba layers
        self.mamba_layers = nn.ModuleList([
            BidirectionalMamba(d_model) for _ in range(n_layers)
        ])

        # MoE (Mixture of Experts) 门控
        self.expert_gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )

        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(4)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: (B, C, T) = (B, 20, 2000)

        # 空间卷积
        x = self.spatial_conv(x)  # (B, D, T)

        # 时间卷积
        x = self.temporal_conv(x)  # (B, D, T)

        # 转换为序列格式
        x = x.permute(0, 2, 1)  # (B, T, D)

        # Bidirectional Mamba layers
        for mamba in self.mamba_layers:
            x = mamba(x)

        # MoE门控
        gate_weights = self.expert_gate(x.mean(dim=1))  # (B, 4)
        expert_outputs = torch.stack([expert(x.mean(dim=1)) for expert in self.experts], dim=1)
        x_moe = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)

        # 输出
        out = self.fc(x_moe)
        out = self.norm(out)

        return out  # (B, out_dim)
```

### 4.2 CMFViT模块 (Vision Transformer + CNN融合)

```python
import timm

class CMFViTEncoder(nn.Module):
    """
    CNN-Mamba-ViT融合编码器
    参考: CMFViT (J Transl Med 2025)
    """

    def __init__(self, in_channels=4, out_dim=256):
        super().__init__()

        # CNN分支 - 局部特征
        self.cnn = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=True,
            in_chans=in_channels,
            num_classes=0
        )
        cnn_dim = self.cnn.num_features

        # ViT分支 - 全局特征
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,
            in_chans=in_channels,
            num_classes=0
        )
        vit_dim = self.vit.num_features

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        # x: (B, 4, H, W)

        # Resize for ViT (224x224)
        x_vit = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # CNN特征
        cnn_feat = self.cnn(x)

        # ViT特征
        vit_feat = self.vit(x_vit)

        # 融合
        combined = torch.cat([cnn_feat, vit_feat], dim=-1)
        out = self.fusion(combined)

        return out  # (B, out_dim)
```

### 4.3 自适应门控融合模块

```python
class AdaptiveGatedFusion(nn.Module):
    """
    自适应门控融合 + 模态质量评估
    参考: AMC (ICML 2025), Hate-UDF (2025)
    """

    def __init__(self, dim=256, n_modalities=3):
        super().__init__()

        self.n_modalities = n_modalities

        # 模态质量评估
        self.quality_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(n_modalities)
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

        # 融合投影
        self.projection = nn.Sequential(
            nn.Linear(dim * n_modalities, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

        # 抗模态坍塌监控
        self.register_buffer('modality_ranks', torch.ones(n_modalities))

    def forward(self, features):
        """
        Args:
            features: list of (B, dim) tensors, one per modality
        """
        batch_size = features[0].shape[0]

        # 计算模态质量权重
        quality_weights = []
        for i, (feat, estimator) in enumerate(zip(features, self.quality_estimators)):
            q = estimator(feat)  # (B, 1)
            quality_weights.append(q)
        quality_weights = torch.cat(quality_weights, dim=-1)  # (B, n_modalities)
        quality_weights = F.softmax(quality_weights, dim=-1)

        # 计算门控权重
        concat_feat = torch.cat(features, dim=-1)  # (B, dim * n_modalities)
        gate_weights = []
        for i, gate in enumerate(self.gates):
            g = gate(concat_feat)  # (B, 1)
            gate_weights.append(g)
        gate_weights = torch.cat(gate_weights, dim=-1)  # (B, n_modalities)

        # 结合质量和门控
        final_weights = quality_weights * gate_weights
        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 加权融合
        weighted_features = []
        for i, feat in enumerate(features):
            w = final_weights[:, i:i+1]  # (B, 1)
            weighted_features.append(feat * w)

        fused = torch.cat(weighted_features, dim=-1)
        out = self.projection(fused)

        # 监控模态有效秩（用于抗模态坍塌）
        if self.training:
            for i, feat in enumerate(features):
                rank = self._compute_effective_rank(feat)
                self.modality_ranks[i] = 0.9 * self.modality_ranks[i] + 0.1 * rank

        return out, final_weights

    def _compute_effective_rank(self, x):
        """计算特征矩阵的有效秩"""
        _, s, _ = torch.svd(x)
        s = s / s.sum()
        entropy = -(s * torch.log(s + 1e-8)).sum()
        return torch.exp(entropy)

    def get_collapse_loss(self):
        """抗模态坍塌损失"""
        # 惩罚有效秩差异过大
        rank_std = self.modality_ranks.std()
        return rank_std
```

### 4.4 完整模型

```python
class HMSModel2025(nn.Module):
    """
    HMS有害脑活动分类模型 - 2025 SOTA架构

    融合技术:
    - EEGMamba: 双向状态空间模型 + MoE
    - CMFViT: Vision Transformer + CNN多流融合
    - 自适应门控融合 + 模态质量评估
    - 监督对比学习
    """

    def __init__(self, num_classes=6, dim=256):
        super().__init__()

        # 模态编码器
        self.eeg_encoder = EEGMambaEncoder(in_channels=20, out_dim=dim)
        self.eeg_spec_encoder = CMFViTEncoder(in_channels=20, out_dim=dim)  # EEG生成的频谱图
        self.spec_encoder = CMFViTEncoder(in_channels=4, out_dim=dim)  # 原始频谱图

        # 自适应融合
        self.fusion = AdaptiveGatedFusion(dim=dim, n_modalities=3)

        # 对比学习投影头
        self.contrastive_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 128)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim // 2, num_classes)
        )

    def forward(self, eeg, eeg_spec, spec, return_features=False):
        """
        Args:
            eeg: (B, 20, 2000) 原始EEG
            eeg_spec: (B, 20, F, T) EEG生成的频谱图
            spec: (B, 4, H, W) 原始频谱图
        """
        # 编码各模态
        eeg_feat = self.eeg_encoder(eeg)
        eeg_spec_feat = self.eeg_spec_encoder(eeg_spec)
        spec_feat = self.spec_encoder(spec)

        # 融合
        fused, weights = self.fusion([eeg_feat, eeg_spec_feat, spec_feat])

        # 分类
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        if return_features:
            contrastive_feat = self.contrastive_head(fused)
            return probs, contrastive_feat, weights

        return probs

    def get_collapse_loss(self):
        return self.fusion.get_collapse_loss()
```

---

## 五、损失函数设计

```python
class HMSLoss(nn.Module):
    """
    HMS综合损失函数

    组成:
    1. KL散度损失 (主任务)
    2. 监督对比学习损失
    3. 抗模态坍塌损失
    4. 标签平滑
    """

    def __init__(self, temp=0.07, alpha_contrastive=0.1, alpha_collapse=0.01):
        super().__init__()
        self.temp = temp
        self.alpha_contrastive = alpha_contrastive
        self.alpha_collapse = alpha_collapse

    def kl_divergence(self, pred, target):
        """KL散度损失"""
        pred = torch.clamp(pred, min=1e-8)
        target = torch.clamp(target, min=1e-8)
        return F.kl_div(pred.log(), target, reduction='batchmean')

    def supervised_contrastive_loss(self, features, labels):
        """监督对比学习损失"""
        # features: (B, D), labels: (B, 6) soft labels

        # 使用argmax作为伪类别
        pseudo_labels = labels.argmax(dim=-1)

        # 计算相似度
        features = F.normalize(features, dim=-1)
        sim = torch.mm(features, features.t()) / self.temp

        # 创建mask
        labels_eq = pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)
        mask = labels_eq.float() - torch.eye(len(labels)).to(labels.device)
        mask = torch.clamp(mask, min=0)

        # 计算损失
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=-1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)

        loss = -mean_log_prob_pos.mean()
        return loss

    def forward(self, pred, target, features=None, collapse_loss=None):
        """
        Args:
            pred: (B, 6) 预测概率
            target: (B, 6) 目标概率分布
            features: (B, D) 对比学习特征 (可选)
            collapse_loss: 模态坍塌损失 (可选)
        """
        # 主损失: KL散度
        loss_kl = self.kl_divergence(pred, target)

        total_loss = loss_kl

        # 对比学习损失
        if features is not None:
            loss_contrastive = self.supervised_contrastive_loss(features, target)
            total_loss = total_loss + self.alpha_contrastive * loss_contrastive

        # 抗模态坍塌损失
        if collapse_loss is not None:
            total_loss = total_loss + self.alpha_collapse * collapse_loss

        return total_loss
```

---

## 六、训练Pipeline

### 6.1 两阶段训练策略（Kaggle获奖方案）

```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class HMSTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['T_0'],
            T_mult=2
        )

        self.criterion = HMSLoss(
            temp=config.get('temp', 0.07),
            alpha_contrastive=config.get('alpha_contrastive', 0.1),
            alpha_collapse=config.get('alpha_collapse', 0.01)
        )

        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, loader, stage='full'):
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f'Training ({stage})')

        for batch in pbar:
            eeg = batch['eeg'].to(self.device)
            eeg_spec = batch['eeg_spec'].to(self.device)
            spec = batch['spec'].to(self.device)
            label = batch['label'].to(self.device)

            with torch.cuda.amp.autocast():
                pred, features, _ = self.model(eeg, eeg_spec, spec, return_features=True)
                collapse_loss = self.model.get_collapse_loss()
                loss = self.criterion(pred, label, features, collapse_loss)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(loader, desc='Validating'):
            eeg = batch['eeg'].to(self.device)
            eeg_spec = batch['eeg_spec'].to(self.device)
            spec = batch['spec'].to(self.device)
            label = batch['label'].to(self.device)

            pred = self.model(eeg, eeg_spec, spec)
            loss = F.kl_div(pred.log(), label, reduction='batchmean')

            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算KL散度
        kl_div = F.kl_div(all_preds.log(), all_labels, reduction='batchmean').item()

        return total_loss / len(loader), kl_div

    def two_stage_training(self, train_df, val_df, eeg_dir, spec_dir):
        """
        两阶段训练策略（参考Kaggle 1st place方案）

        Stage 1: 高置信度样本预训练
        Stage 2: 全数据微调
        """
        print("="*60)
        print("Stage 1: 高置信度样本预训练")
        print("="*60)

        # 筛选高置信度样本 (投票一致性高的样本)
        vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        train_df['max_vote_ratio'] = train_df[vote_cols].max(axis=1) / train_df[vote_cols].sum(axis=1)
        high_conf_df = train_df[train_df['max_vote_ratio'] > 0.7]

        print(f"高置信度样本: {len(high_conf_df)} / {len(train_df)} ({100*len(high_conf_df)/len(train_df):.1f}%)")

        # Stage 1 训练
        high_conf_dataset = HMSDataset(high_conf_df, eeg_dir, spec_dir, mode='train')
        high_conf_loader = DataLoader(
            high_conf_dataset,
            batch_size=self.config['batch_size'],
            sampler=create_patient_balanced_sampler(high_conf_df),
            num_workers=4
        )

        for epoch in range(self.config['stage1_epochs']):
            train_loss = self.train_epoch(high_conf_loader, stage='stage1')
            print(f"Stage 1 Epoch {epoch+1}: Loss = {train_loss:.4f}")

        print("\n" + "="*60)
        print("Stage 2: 全数据微调")
        print("="*60)

        # 降低学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config['lr'] * 0.1

        # Stage 2 训练
        full_dataset = HMSDataset(train_df, eeg_dir, spec_dir, mode='train')
        val_dataset = HMSDataset(val_df, eeg_dir, spec_dir, mode='val')

        full_loader = DataLoader(
            full_dataset,
            batch_size=self.config['batch_size'],
            sampler=create_patient_balanced_sampler(train_df),
            num_workers=4
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=4)

        best_kl = float('inf')
        for epoch in range(self.config['stage2_epochs']):
            train_loss = self.train_epoch(full_loader, stage='stage2')
            val_loss, val_kl = self.validate(val_loader)

            print(f"Stage 2 Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val KL = {val_kl:.4f}")

            if val_kl < best_kl:
                best_kl = val_kl
                torch.save(self.model.state_dict(), f'models/best_model.pth')
                print(f"  → Saved best model with KL = {val_kl:.4f}")

        return best_kl
```

### 6.2 完整训练脚本

```python
def main():
    # 配置
    config = {
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_folds': 5,
        'batch_size': 32,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'T_0': 500,
        'stage1_epochs': 5,
        'stage2_epochs': 15,
        'temp': 0.07,
        'alpha_contrastive': 0.1,
        'alpha_collapse': 0.01
    }

    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # 加载数据
    df = pd.read_csv('data/train.csv')
    eeg_dir = 'data/train_eegs'
    spec_dir = 'data/train_spectrograms'

    # 创建无泄露的划分
    folds = create_leak_free_folds(df, n_splits=config['n_folds'])

    # 存储每个fold的结果
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'#'*60}")
        print(f"# Fold {fold + 1}/{config['n_folds']}")
        print(f"{'#'*60}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        print(f"训练集: {len(train_df)} 样本, {train_df['patient_id'].nunique()} 患者")
        print(f"验证集: {len(val_df)} 样本, {val_df['patient_id'].nunique()} 患者")

        # 创建模型
        model = HMSModel2025(num_classes=6, dim=256).to(config['device'])

        # 训练
        trainer = HMSTrainer(model, config)
        best_kl = trainer.two_stage_training(train_df, val_df, eeg_dir, spec_dir)

        fold_results.append(best_kl)
        print(f"\nFold {fold+1} Best KL: {best_kl:.4f}")

        # 保存fold模型
        torch.save(model.state_dict(), f'models/fold{fold}_best.pth')

    # 汇总结果
    print(f"\n{'='*60}")
    print("交叉验证结果汇总")
    print(f"{'='*60}")
    for i, kl in enumerate(fold_results):
        print(f"Fold {i+1}: KL = {kl:.4f}")
    print(f"平均 KL: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")


if __name__ == '__main__':
    main()
```

---

## 七、推理与集成

```python
def inference_with_tta(models, loader, device, n_tta=5):
    """
    Test Time Augmentation推理

    TTA策略:
    1. 原始数据
    2. 水平翻转频谱图
    3. 时间偏移EEG
    4. 添加轻微噪声
    5. 不同窗口偏移
    """
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            eeg = batch['eeg'].to(device)
            eeg_spec = batch['eeg_spec'].to(device)
            spec = batch['spec'].to(device)

            tta_preds = []

            # 原始预测
            for model in models:
                model.eval()
                pred = model(eeg, eeg_spec, spec)
                tta_preds.append(pred)

            # TTA: 水平翻转
            for model in models:
                pred = model(eeg, eeg_spec, torch.flip(spec, dims=[-1]))
                tta_preds.append(pred)

            # TTA: 时间翻转
            for model in models:
                pred = model(torch.flip(eeg, dims=[-1]), eeg_spec, spec)
                tta_preds.append(pred)

            # 平均所有TTA预测
            avg_pred = torch.stack(tta_preds).mean(dim=0)
            all_preds.append(avg_pred.cpu())

    return torch.cat(all_preds, dim=0)


def ensemble_models(model_paths, test_loader, device):
    """模型集成"""
    models = []
    for path in model_paths:
        model = HMSModel2025(num_classes=6, dim=256).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)

    # TTA推理
    predictions = inference_with_tta(models, test_loader, device)

    return predictions
```

---

## 八、预期性能与资源需求

### 8.1 预期性能

| 方案 | 预期KL-Div | 说明 |
|------|------------|------|
| 单模型基线 | ~0.35 | 仅EfficientNet |
| 多模态融合 | ~0.30 | EEG + Spec |
| + 两阶段训练 | ~0.28 | 高置信度预训练 |
| + TTA + 集成 | ~0.27 | 5-fold + TTA |
| **目标** | **<0.275** | 接近1st place |

### 8.2 计算资源需求

| 资源 | 推荐配置 |
|------|----------|
| GPU | NVIDIA RTX 3090 / A100 |
| GPU显存 | 24GB+ (混合精度训练可用16GB) |
| 内存 | 64GB+ |
| 存储 | 100GB+ SSD |
| 训练时间 | 约8-12小时/fold |

### 8.3 依赖包

```bash
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
mamba-ssm>=1.2.0  # 可选，需要CUDA
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
albumentations>=1.3.0
tqdm>=4.65.0
```

---

## 九、项目目录结构

```
HMS_Project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_eegs/
│   ├── train_spectrograms/
│   ├── test_eegs/
│   └── test_spectrograms/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── processor.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── eeg_mamba.py
│   │   ├── cmfvit.py
│   │   ├── fusion.py
│   │   └── model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── scheduler.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── configs/
│   └── config.yaml
├── models/           # 保存的模型权重
├── notebooks/
│   ├── eda.ipynb
│   └── analysis.ipynb
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## 十、参考文献

### 2025年SOTA模型

1. [FEMBA: Efficient and Scalable EEG Analysis with a Bidirectional Mamba Foundation Model](https://arxiv.org/abs/2502.06438) - arXiv 2025
2. [EEGMamba: Bidirectional State Space Model with Mixture of Experts](https://arxiv.org/abs/2407.20254) - ICLR 2025
3. [CMFViT: Multi-stream feature fusion of vision transformer and CNN](https://link.springer.com/article/10.1186/s12967-025-06862-z) - J Transl Med 2025
4. [Large Cognition Model: Pretrained EEG Foundation Model](https://arxiv.org/abs/2502.17464) - arXiv 2025

### 预训练模型

5. [LaBraM: Large Brain Model for Learning Generic Representations](https://github.com/935963004/LaBraM) - ICLR 2024 Spotlight
6. [EEGPT: Pretrained Transformer for Universal EEG Representation](https://openreview.net/forum?id=lvS2b8CjG5) - ICLR 2024

### Kaggle方案

7. [HMS 1st Place Solution - Team Sony](https://www.kaggle.com/code/hoyso48/1st-place-solution-training)
8. [HMS 7th Place Solution](https://github.com/gunesevitan/hms-harmful-brain-activity-classification)
9. [HMS 15th Place Solution](https://github.com/brendanartley/HMS-Competition)
