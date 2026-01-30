# HMS - Harmful Brain Activity Classification 实验方案

## 一、比赛概述

| 项目 | 内容 |
|------|------|
| **任务目标** | 预测专家对EEG信号中6类有害脑活动的投票概率分布 |
| **评价指标** | KL散度 (KL-Divergence) - 越小越好 |
| **第1名成绩** | 0.272332 (team Sony) |
| **数据规模** | 106,800样本，17,089个EEG，1,950名患者 |

---

## 二、数据结构分析

```
数据集结构：
├── train.csv (106,800 samples)
├── train_eegs/ (17,301 parquet files)
│   └── EEG: 20通道 × 10,000时间步 (50秒@200Hz)
├── train_spectrograms/ (11,138 parquet files)
│   └── Spectrogram: 4区域(LL,LP,RL,RP) × 100频率 × 300时间步
└── 6类标签: Seizure, LPD, GPD, LRDA, GRDA, Other
```

### EEG通道 (20通道)
```
Fp1, F3, C3, P3, F7, T3, T5, O1, Fz, Cz,
Pz, Fp2, F4, C4, P4, F8, T4, T6, O2, EKG
```

### 类别分布
| 类别 | 样本数 | 比例 |
|------|--------|------|
| Seizure | 20,933 | 19.6% |
| GRDA | 18,861 | 17.7% |
| Other | 18,808 | 17.6% |
| GPD | 16,702 | 15.6% |
| LRDA | 16,640 | 15.6% |
| LPD | 14,856 | 13.9% |

### 标签含义
- **Seizure**: 癫痫发作
- **LPD (Lateralized Periodic Discharges)**: 侧向周期性放电
- **GPD (Generalized Periodic Discharges)**: 广泛性周期性放电
- **LRDA (Lateralized Rhythmic Delta Activity)**: 侧向节律性δ活动
- **GRDA (Generalized Rhythmic Delta Activity)**: 广泛性节律性δ活动
- **Other**: 其他

---

## 三、2025年SOTA方法与推荐模型

### 3.1 推荐模型架构

| 模型 | 类型 | 适用场景 | 来源 |
|------|------|----------|------|
| **EEGPT** | 预训练Transformer | EEG通用特征提取 | ICLR 2024 |
| **CNN-Informer** | CNN+Transformer混合 | 长序列EEG依赖建模 | Neural Networks 2024 |
| **CMFViT** | ViT+CNN多流融合 | 频谱图图像分类 | J Transl Med 2025 |
| **EfficientNetV2** | CNN | 频谱图图像backbone | Kaggle获奖方案 |
| **1D-WaveNet** | 1D CNN | 原始EEG波形处理 | Kaggle获奖方案 |
| **AFTA-Transformer** | 自监督Transformer | 少标签场景 | MDPI 2025 |

### 3.2 Kaggle比赛获奖方案核心技术

```
第1名 (team Sony): KL-Div = 0.272332
├── 多模态融合 (EEG + Spectrogram)
├── EfficientNet系列 (图像backbone)
├── WaveNet (1D EEG处理)
└── CatBoost (特征融合后的集成)

第7名方案:
├── EfficientNetV2 + ResNet ensemble
├── Spectrogram预处理与增强
└── 多折交叉验证 + TTA

第15名方案:
├── 1D-WaveNet (改进版)
├── 可训练STFT层
├── HGNetV2-B4 + EfficientNetV2-S
└── Attention Pooling
```

---

## 四、推荐实验方案

### 方案A: 多模态双分支架构 (推荐)

```
                    输入
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    [EEG分支]              [Spectrogram分支]
    20ch × 10000           4区×100freq×300t
         │                       │
         ▼                       ▼
   ┌─────────────┐        ┌─────────────┐
   │ 1D-WaveNet  │        │EfficientNet │
   │ or EEGPT    │        │  V2-S/B4    │
   └─────────────┘        └─────────────┘
         │                       │
         ▼                       ▼
    [768-dim]               [512-dim]
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌────────────────┐
            │  Cross-Modal   │
            │   Attention    │
            └────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │   Classifier   │
            │   (6 classes)  │
            └────────────────┘
                     │
                     ▼
            Softmax → 投票概率分布
```

### 方案B: 纯Spectrogram图像方案 (简化版)

```python
# 伪代码
class SpectrogramModel(nn.Module):
    def __init__(self):
        self.backbone = EfficientNetV2_S(pretrained=True)
        self.head = nn.Linear(1280, 6)

    def forward(self, spec_images):
        # spec_images: [B, 4, H, W] -> 4个脑区频谱图
        features = self.backbone(spec_images)
        logits = self.head(features)
        return F.softmax(logits, dim=-1)
```

### 方案C: 2025最新 - EEGPT预训练 + 微调

```python
# 使用EEGPT预训练模型
class EEGPTModel(nn.Module):
    def __init__(self):
        self.eegpt = EEGPT.from_pretrained('eegpt-base')  # 10M参数
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )

    def forward(self, eeg):
        features = self.eegpt(eeg)  # [B, 768]
        logits = self.classifier(features)
        return F.softmax(logits, dim=-1)
```

---

## 五、详细实现步骤

### 5.1 数据预处理

#### EEG预处理
```python
import numpy as np
from scipy import signal

def preprocess_eeg(eeg_data, fs=200):
    """
    EEG预处理流程
    Args:
        eeg_data: numpy array, shape (time_steps, channels)
        fs: 采样率, 默认200Hz
    """
    # 1. 带通滤波: 0.5-40Hz (去除基线漂移和高频噪声)
    b, a = signal.butter(4, [0.5, 40], btype='band', fs=fs)
    filtered = signal.filtfilt(b, a, eeg_data, axis=0)

    # 2. Z-score标准化 (per channel)
    mean = np.mean(filtered, axis=0, keepdims=True)
    std = np.std(filtered, axis=0, keepdims=True) + 1e-8
    normalized = (filtered - mean) / std

    # 3. 裁剪异常值
    normalized = np.clip(normalized, -10, 10)

    return normalized

def extract_eeg_window(eeg_data, offset_sec, window_sec=10, fs=200):
    """
    提取指定时间窗口的EEG数据
    """
    start = int(offset_sec * fs)
    end = start + int(window_sec * fs)
    return eeg_data[start:end]
```

#### Spectrogram预处理
```python
def preprocess_spectrogram(spec_data):
    """
    Spectrogram预处理流程
    Args:
        spec_data: DataFrame, 包含LL, LP, RL, RP四个区域的频谱数据
    """
    # 1. 填充NaN值
    spec_data = spec_data.fillna(0)

    # 2. 分离四个脑区
    regions = ['LL', 'LP', 'RL', 'RP']
    images = []
    for region in regions:
        cols = [c for c in spec_data.columns if c.startswith(region)]
        region_data = spec_data[cols].values
        images.append(region_data)

    # 3. Stack为4通道图像: (4, freq, time)
    spec_image = np.stack(images, axis=0)

    # 4. Log变换压缩动态范围
    spec_image = np.log1p(np.clip(spec_image, 0, None))

    # 5. 归一化到 [0, 1]
    spec_image = (spec_image - spec_image.min()) / (spec_image.max() - spec_image.min() + 1e-8)

    return spec_image.astype(np.float32)
```

### 5.2 数据增强

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# EEG增强
class EEGAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, eeg):
        if np.random.rand() < self.p:
            # 时间偏移 (±2秒)
            shift = np.random.randint(-400, 400)
            eeg = np.roll(eeg, shift, axis=0)

        if np.random.rand() < self.p:
            # 高斯噪声
            noise = np.random.normal(0, 0.01, eeg.shape)
            eeg = eeg + noise

        if np.random.rand() < self.p:
            # 通道Dropout (随机mask 10%通道)
            n_drop = max(1, int(eeg.shape[1] * 0.1))
            drop_idx = np.random.choice(eeg.shape[1], n_drop, replace=False)
            eeg[:, drop_idx] = 0

        return eeg

# Spectrogram增强
spec_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2()
])
```

### 5.3 Dataset类

```python
import torch
from torch.utils.data import Dataset
import pandas as pd

class HMSDataset(Dataset):
    def __init__(self, df, eeg_dir, spec_dir, transform=None, mode='train'):
        self.df = df
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.transform = transform
        self.mode = mode

        # 目标类别
        self.target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote',
                           'lrda_vote', 'grda_vote', 'other_vote']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 加载EEG
        eeg_path = f"{self.eeg_dir}/{row['eeg_id']}.parquet"
        eeg_df = pd.read_parquet(eeg_path)
        eeg = eeg_df.values  # (10000, 20)

        # 提取窗口
        offset = row['eeg_label_offset_seconds']
        eeg = extract_eeg_window(eeg, offset)
        eeg = preprocess_eeg(eeg)

        # 加载Spectrogram
        spec_path = f"{self.spec_dir}/{row['spectrogram_id']}.parquet"
        spec_df = pd.read_parquet(spec_path)
        spec = preprocess_spectrogram(spec_df)

        # 数据增强
        if self.transform and self.mode == 'train':
            eeg = self.transform['eeg'](eeg)
            spec = self.transform['spec'](image=spec.transpose(1,2,0))['image']

        # 标签 (软标签 - 投票概率分布)
        if self.mode == 'train':
            votes = row[self.target_cols].values.astype(np.float32)
            label = votes / (votes.sum() + 1e-8)  # 归一化
        else:
            label = np.zeros(6, dtype=np.float32)

        return {
            'eeg': torch.tensor(eeg, dtype=torch.float32).permute(1, 0),  # (20, 2000)
            'spec': torch.tensor(spec, dtype=torch.float32),  # (4, H, W)
            'label': torch.tensor(label, dtype=torch.float32)
        }
```

### 5.4 模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class WaveNetBlock(nn.Module):
    """1D WaveNet Block for EEG processing"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size-1)*dilation//2, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class EEGEncoder(nn.Module):
    """1D WaveNet-based EEG Encoder"""
    def __init__(self, in_channels=20, hidden_dim=128, out_dim=256):
        super().__init__()

        self.blocks = nn.Sequential(
            WaveNetBlock(in_channels, hidden_dim, dilation=1),
            WaveNetBlock(hidden_dim, hidden_dim, dilation=2),
            WaveNetBlock(hidden_dim, hidden_dim, dilation=4),
            WaveNetBlock(hidden_dim, hidden_dim, dilation=8),
            WaveNetBlock(hidden_dim, hidden_dim, dilation=16),
            WaveNetBlock(hidden_dim, out_dim, dilation=32),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, 20, T)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)  # (B, out_dim)
        return x

class SpectrogramEncoder(nn.Module):
    """EfficientNet-based Spectrogram Encoder"""
    def __init__(self, model_name='tf_efficientnetv2_s', pretrained=True, out_dim=256):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          in_chans=4, num_classes=0)
        backbone_dim = self.backbone.num_features
        self.fc = nn.Linear(backbone_dim, out_dim)

    def forward(self, x):
        # x: (B, 4, H, W)
        x = self.backbone(x)  # (B, backbone_dim)
        x = self.fc(x)  # (B, out_dim)
        return x

class CrossModalAttention(nn.Module):
    """Cross-Modal Attention for EEG-Spectrogram Fusion"""
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, eeg_feat, spec_feat):
        # eeg_feat, spec_feat: (B, dim)
        eeg_feat = eeg_feat.unsqueeze(1)  # (B, 1, dim)
        spec_feat = spec_feat.unsqueeze(1)  # (B, 1, dim)

        # Cross attention: EEG queries Spectrogram
        combined = torch.cat([eeg_feat, spec_feat], dim=1)  # (B, 2, dim)
        attn_out, _ = self.attn(combined, combined, combined)
        attn_out = self.norm(attn_out + combined)

        return attn_out.mean(dim=1)  # (B, dim)

class HMSModel(nn.Module):
    """Multi-modal HMS Classification Model"""
    def __init__(self, num_classes=6, eeg_dim=256, spec_dim=256):
        super().__init__()

        self.eeg_encoder = EEGEncoder(out_dim=eeg_dim)
        self.spec_encoder = SpectrogramEncoder(out_dim=spec_dim)
        self.fusion = CrossModalAttention(dim=eeg_dim)

        self.classifier = nn.Sequential(
            nn.Linear(eeg_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, eeg, spec):
        eeg_feat = self.eeg_encoder(eeg)
        spec_feat = self.spec_encoder(spec)
        fused = self.fusion(eeg_feat, spec_feat)
        logits = self.classifier(fused)
        return F.softmax(logits, dim=-1)
```

### 5.5 训练代码

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

def kl_divergence_loss(pred, target):
    """KL散度损失函数"""
    pred = torch.clamp(pred, min=1e-8)
    target = torch.clamp(target, min=1e-8)
    return F.kl_div(pred.log(), target, reduction='batchmean')

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        eeg = batch['eeg'].to(device)
        spec = batch['spec'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()
        pred = model(eeg, spec)
        loss = kl_divergence_loss(pred, label)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            eeg = batch['eeg'].to(device)
            spec = batch['spec'].to(device)
            label = batch['label'].to(device)

            pred = model(eeg, spec)
            loss = kl_divergence_loss(pred, label)
            total_loss += loss.item()

    return total_loss / len(loader)

def main():
    # 配置
    config = {
        'seed': 42,
        'n_folds': 5,
        'epochs': 20,
        'batch_size': 32,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 加载数据
    train_df = pd.read_csv('data/train.csv')

    # GroupKFold (按patient_id分组)
    gkf = GroupKFold(n_splits=config['n_folds'])

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=train_df['patient_id'])):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{config['n_folds']}")
        print(f"{'='*50}")

        train_data = train_df.iloc[train_idx]
        val_data = train_df.iloc[val_idx]

        train_dataset = HMSDataset(train_data, 'data/train_eegs', 'data/train_spectrograms', mode='train')
        val_dataset = HMSDataset(val_data, 'data/train_eegs', 'data/train_spectrograms', mode='val')

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

        # 模型
        model = HMSModel().to(config['device'])

        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), T_mult=2)

        # 训练
        best_val_loss = float('inf')
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")

            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, config['device'])
            val_loss = validate(model, val_loader, config['device'])

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'models/fold{fold}_best.pth')
                print(f"Saved best model with val_loss: {val_loss:.4f}")

if __name__ == '__main__':
    main()
```

---

## 六、推理与集成

```python
def inference_with_tta(model, loader, device, n_tta=3):
    """带TTA的推理"""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            eeg = batch['eeg'].to(device)
            spec = batch['spec'].to(device)

            preds = []
            # 原始预测
            preds.append(model(eeg, spec))

            # TTA: 水平翻转
            preds.append(model(eeg, torch.flip(spec, dims=[-1])))

            # TTA: 时间翻转
            preds.append(model(torch.flip(eeg, dims=[-1]), spec))

            # 平均
            pred = torch.stack(preds).mean(dim=0)
            all_preds.append(pred.cpu())

    return torch.cat(all_preds, dim=0)

def ensemble_predictions(fold_predictions, weights=None):
    """
    集成多个fold的预测
    Args:
        fold_predictions: list of predictions, each shape (N, 6)
        weights: optional weights for each fold
    """
    if weights is None:
        weights = [1.0] * len(fold_predictions)

    weighted_preds = sum(w * p for w, p in zip(weights, fold_predictions))
    return weighted_preds / sum(weights)
```

---

## 七、代码框架建议

```
project/
├── configs/
│   └── config.yaml
├── data/
│   ├── train.csv
│   ├── train_eegs/
│   ├── train_spectrograms/
│   ├── test_eegs/
│   └── test_spectrograms/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py      # Dataset类
│   │   ├── transforms.py   # 数据增强
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── eeg_encoder.py  # WaveNet/EEGPT
│   │   ├── spec_encoder.py # EfficientNet
│   │   ├── fusion.py       # 融合模块
│   │   └── model.py        # 完整模型
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── losses.py       # KL-Div Loss
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── notebooks/
│   └── eda.ipynb
├── models/              # 保存训练好的模型
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## 八、预期性能与资源估算

| 方案 | 预期KL-Div | 模型大小 | 训练时间(单fold) |
|------|------------|----------|------------------|
| 单EfficientNet | ~0.35 | ~20M | ~2h |
| EfficientNet + WaveNet | ~0.30 | ~40M | ~4h |
| 多模态融合 + 集成 | ~0.28 | ~100M | ~8h |
| SOTA (team Sony) | 0.272 | - | - |

---

## 九、2025年最新技术建议

### 9.1 EEGPT预训练模型
- **论文**: EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals
- **代码**: https://github.com/BINE022/EEGPT
- **特点**: 10M参数，mask-based双自监督学习，时空表示对齐
- **用法**: 直接加载预训练权重，在HMS数据上微调

### 9.2 CNN-Informer混合架构
- **论文**: CNN-Informer: A hybrid deep learning model for seizure detection on long-term EEG
- **特点**: CNN提取局部特征 + Informer捕捉长程依赖，低计算复杂度
- **适用**: 处理完整50秒EEG序列

### 9.3 自监督预训练 (AFTA-Transformer)
- **论文**: Self-Supervised Learning with Adaptive Frequency-Time Attention Transformer
- **特点**: mask-and-reconstruction预训练，自适应频率-时间注意力
- **成绩**: 在TUSZ, TUAB, TUEV数据集上SOTA

### 9.4 多流融合 (CMFViT)
- **论文**: Multi-stream feature fusion of vision transformer and CNN for precise epileptic seizure detection
- **特点**: ViT捕捉全局信息 + CNN提取局部特征，多流融合策略
- **发表**: Journal of Translational Medicine, 2025

---

## 十、参考资源

### 论文
1. [Deep learning in intracranial EEG for seizure detection (Frontiers 2025)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1677898/full)
2. [Multi-stream feature fusion of ViT and CNN (J Transl Med 2025)](https://link.springer.com/article/10.1186/s12967-025-06862-z)
3. [Transformer-based EEG Decoding Survey (arXiv 2025)](https://arxiv.org/html/2507.02320v1)
4. [CNN-Informer (Neural Networks 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0893608024007792)
5. [EEGPT (ICLR 2024)](https://openreview.net/forum?id=lvS2b8CjG5)

### Kaggle方案
1. [HMS 7th Place Solution (GitHub)](https://github.com/gunesevitan/hms-harmful-brain-activity-classification)
2. [HMS 15th Place Solution (GitHub)](https://github.com/brendanartley/HMS-Competition)
3. [HMS Competition Discussion](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion)

### 代码库
1. [EEGPT GitHub](https://github.com/BINE022/EEGPT)
2. [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
3. [EEGNet](https://github.com/vlawhern/arl-eegmodels)
