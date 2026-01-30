# HMS 项目 Google Colab 实验指南

## 目录
1. [快速开始](#1-快速开始)
2. [环境配置](#2-环境配置)
3. [数据准备](#3-数据准备)
4. [运行训练](#4-运行训练)
5. [查看结果](#5-查看结果)
6. [常见问题](#6-常见问题)
7. [完整Notebook代码](#7-完整notebook代码)

---

## 1. 快速开始

### 1.1 创建新Notebook
1. 访问 [Google Colab](https://colab.research.google.com/)
2. 点击 "新建笔记本"
3. 修改运行时类型：`运行时` → `更改运行时类型` → 选择 `GPU` (推荐T4或更高)

### 1.2 检查GPU
```python
# 检查GPU是否可用
!nvidia-smi
```

---

## 2. 环境配置

### 2.1 安装依赖
```python
# 安装PyTorch (Colab通常已预装)
!pip install torch torchvision torchaudio --quiet

# 安装项目依赖
!pip install timm einops pyyaml tqdm matplotlib seaborn scikit-learn pandas --quiet

# 安装Kaggle API
!pip install kaggle --quiet

# 可选：安装Mamba（需要CUDA 11.6+）
# !pip install mamba-ssm --no-build-isolation --quiet

print("✓ 依赖安装完成!")
```

### 2.2 验证安装
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## 3. 数据准备

### 3.1 配置Kaggle API

**方法一：上传kaggle.json（推荐）**
```python
# 1. 从Kaggle下载API密钥
# 访问 https://www.kaggle.com/settings → Create New Token → 下载kaggle.json

# 2. 在Colab中上传
from google.colab import files
uploaded = files.upload()  # 选择kaggle.json文件

# 3. 配置Kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
print("✓ Kaggle API配置完成!")
```

**方法二：直接输入密钥**
```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'  # 替换为你的用户名
os.environ['KAGGLE_KEY'] = 'your_api_key'        # 替换为你的API密钥
```

### 3.2 下载比赛数据
```python
# 创建数据目录
!mkdir -p /content/data

# 下载HMS比赛数据（约50GB，需要较长时间）
!kaggle competitions download -c hms-harmful-brain-activity-classification -p /content/data

# 解压数据
!cd /content/data && unzip -q '*.zip'

# 查看数据结构
!ls -la /content/data/
```

### 3.3 使用Google Drive持久化存储（推荐）
```python
# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 数据存储到Drive（避免重复下载）
DATA_DIR = '/content/drive/MyDrive/HMS_Data'

import os
if not os.path.exists(f'{DATA_DIR}/train.csv'):
    !mkdir -p {DATA_DIR}
    !kaggle competitions download -c hms-harmful-brain-activity-classification -p {DATA_DIR}
    !cd {DATA_DIR} && unzip -q '*.zip'
    print("✓ 数据下载并解压完成!")
else:
    print("✓ 数据已存在，跳过下载")

# 验证数据
!ls {DATA_DIR}/
```

---

## 4. 运行训练

### 4.1 克隆项目代码
```python
# 从GitHub克隆（如果你把代码上传到了GitHub）
# !git clone https://github.com/your_username/HMS_Project.git
# %cd HMS_Project

# 或者直接在Colab中创建代码
!mkdir -p /content/HMS_Project
%cd /content/HMS_Project
```

### 4.2 创建项目结构和代码
```python
%%writefile /content/HMS_Project/train_colab.py
"""
HMS Colab训练脚本 - 简化版
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import json
from datetime import datetime

# ============== 配置 ==============
class Config:
    # 数据路径（使用Google Drive）
    data_dir = '/content/drive/MyDrive/HMS_Data'
    train_csv = f'{data_dir}/train.csv'
    eeg_dir = f'{data_dir}/train_eegs'
    spec_dir = f'{data_dir}/train_spectrograms'

    # 模型参数
    num_classes = 6
    hidden_dim = 128  # Colab显存有限，用较小的维度

    # 训练参数
    batch_size = 16  # Colab T4建议16-32
    epochs = 10
    lr = 1e-4
    num_workers = 2

    # 其他
    seed = 42
    fold = 0
    n_folds = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 输出
    output_dir = '/content/drive/MyDrive/HMS_Output'

config = Config()

# ============== 数据集 ==============
class HMSDatasetSimple(Dataset):
    """简化版HMS数据集"""

    def __init__(self, df, eeg_dir, spec_dir, mode='train'):
        self.df = df.reset_index(drop=True)
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.mode = mode

        self.label_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote',
                          'lrda_vote', 'grda_vote', 'other_vote']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 加载EEG
        eeg_path = os.path.join(self.eeg_dir, f"{int(row['eeg_id'])}.parquet")
        try:
            eeg_df = pd.read_parquet(eeg_path)
            eeg = eeg_df.values[:2000, :20].T  # (20, 2000)
            eeg = np.nan_to_num(eeg, nan=0.0)
            # 标准化
            eeg = (eeg - eeg.mean()) / (eeg.std() + 1e-8)
        except:
            eeg = np.zeros((20, 2000))

        # 加载频谱图
        spec_path = os.path.join(self.spec_dir, f"{int(row['spectrogram_id'])}.parquet")
        try:
            spec_df = pd.read_parquet(spec_path)
            spec = spec_df.values[:, 1:]  # 去掉时间列
            # 调整大小为 (128, 256)
            from scipy.ndimage import zoom
            h, w = spec.shape
            spec = zoom(spec, (128/h, 256/w), order=1)
            spec = np.nan_to_num(spec, nan=0.0)
            # 标准化
            spec = (spec - spec.mean()) / (spec.std() + 1e-8)
            spec = np.stack([spec] * 4, axis=0)  # (4, 128, 256)
        except:
            spec = np.zeros((4, 128, 256))

        # 标签
        votes = row[self.label_cols].values.astype(np.float32)
        label = votes / (votes.sum() + 1e-8)

        return {
            'eeg': torch.tensor(eeg, dtype=torch.float32),
            'spec': torch.tensor(spec, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

# ============== 模型 ==============
class SimpleEEGEncoder(nn.Module):
    """简化版EEG编码器"""
    def __init__(self, in_channels=20, hidden_dim=128, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

class SimpleSpecEncoder(nn.Module):
    """简化版频谱图编码器"""
    def __init__(self, in_channels=4, hidden_dim=128, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class HMSModelSimple(nn.Module):
    """简化版HMS模型"""
    def __init__(self, num_classes=6, hidden_dim=128):
        super().__init__()
        self.eeg_encoder = SimpleEEGEncoder(20, hidden_dim, hidden_dim)
        self.spec_encoder = SimpleSpecEncoder(4, hidden_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, eeg, spec):
        eeg_feat = self.eeg_encoder(eeg)
        spec_feat = self.spec_encoder(spec)

        combined = torch.cat([eeg_feat, spec_feat], dim=-1)
        fused = self.fusion(combined)

        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        return {'probs': probs, 'logits': logits}

# ============== 训练函数 ==============
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        eeg = batch['eeg'].to(device)
        spec = batch['spec'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()
        output = model(eeg, spec)

        # KL散度损失
        loss = F.kl_div(output['probs'].log(), label, reduction='batchmean')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc='Validation'):
        eeg = batch['eeg'].to(device)
        spec = batch['spec'].to(device)
        label = batch['label'].to(device)

        output = model(eeg, spec)
        all_preds.append(output['probs'].cpu())
        all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    kl = F.kl_div(all_preds.log(), all_labels, reduction='batchmean').item()

    return kl, all_preds.numpy(), all_labels.numpy()

# ============== 主函数 ==============
def main():
    print("="*60)
    print("HMS Training on Colab")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 加载数据
    print("\nLoading data...")
    df = pd.read_csv(config.train_csv)
    print(f"Total samples: {len(df)}")

    # 数据划分
    gkf = GroupKFold(n_splits=config.n_folds)
    patient_ids = df['patient_id'].values

    for i, (train_idx, val_idx) in enumerate(gkf.split(df, groups=patient_ids)):
        if i == config.fold:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            break

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # 创建数据集
    train_dataset = HMSDatasetSimple(train_df, config.eeg_dir, config.spec_dir, 'train')
    val_dataset = HMSDatasetSimple(val_df, config.eeg_dir, config.spec_dir, 'val')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers)

    # 创建模型
    print("\nCreating model...")
    model = HMSModelSimple(config.num_classes, config.hidden_dim)
    model = model.to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 训练
    history = {'train_loss': [], 'val_kl': [], 'lr': []}
    best_kl = float('inf')

    print("\nStarting training...")
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")

        train_loss = train_epoch(model, train_loader, optimizer, config.device)
        val_kl, preds, labels = validate(model, val_loader, config.device)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_kl'].append(val_kl)
        history['lr'].append(lr)

        print(f"Train Loss: {train_loss:.4f}, Val KL: {val_kl:.4f}, LR: {lr:.2e}")

        # 保存最佳模型
        if val_kl < best_kl:
            best_kl = val_kl
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_kl': val_kl
            }, f'{config.output_dir}/best_model.pth')
            print(f"  -> Saved best model! KL: {val_kl:.4f}")

    # 保存训练历史
    with open(f'{config.output_dir}/history.json', 'w') as f:
        json.dump(history, f)

    # 绘制曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'])
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')

    axes[1].plot(history['val_kl'])
    axes[1].set_title('Validation KL')
    axes[1].set_xlabel('Epoch')
    axes[1].axhline(y=best_kl, color='r', linestyle='--', label=f'Best: {best_kl:.4f}')
    axes[1].legend()

    axes[2].plot(history['lr'])
    axes[2].set_title('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{config.output_dir}/training_curves.png', dpi=150)
    plt.show()

    print("\n" + "="*60)
    print(f"Training completed! Best Val KL: {best_kl:.4f}")
    print(f"Results saved to: {config.output_dir}")
    print("="*60)

    return history, best_kl

if __name__ == "__main__":
    history, best_kl = main()
```

### 4.3 运行训练
```python
# 运行训练脚本
%run /content/HMS_Project/train_colab.py
```

---

## 5. 查看结果

### 5.1 查看训练曲线
```python
from IPython.display import Image
Image('/content/drive/MyDrive/HMS_Output/training_curves.png')
```

### 5.2 加载最佳模型进行推理
```python
# 加载模型
checkpoint = torch.load('/content/drive/MyDrive/HMS_Output/best_model.pth')
print(f"Best model from epoch {checkpoint['epoch']}, Val KL: {checkpoint['val_kl']:.4f}")
```

### 5.3 下载结果到本地
```python
from google.colab import files

# 下载模型
files.download('/content/drive/MyDrive/HMS_Output/best_model.pth')

# 下载训练曲线
files.download('/content/drive/MyDrive/HMS_Output/training_curves.png')
```

---

## 6. 常见问题

### Q1: GPU显存不足 (CUDA Out of Memory)
```python
# 解决方案1：减小batch_size
config.batch_size = 8  # 或更小

# 解决方案2：使用梯度累积
accumulation_steps = 4
# 修改训练循环，每4步才optimizer.step()

# 解决方案3：使用混合精度
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Q2: 运行时断开连接
```python
# 使用Google Drive保存检查点，断开后可以继续训练
# 在训练循环中每隔几个epoch保存检查点

# 恢复训练
checkpoint = torch.load(f'{config.output_dir}/checkpoint_epoch5.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Q3: 数据加载太慢
```python
# 解决方案1：减少num_workers
config.num_workers = 0  # Colab上有时0反而更快

# 解决方案2：预处理数据保存为tensor
# 第一次运行时保存处理后的数据
torch.save(processed_data, '/content/drive/MyDrive/HMS_Data/processed.pt')
```

### Q4: 防止Colab断开
```python
# 在另一个cell运行以下代码保持连接
import time
from IPython.display import clear_output

def keep_alive():
    while True:
        clear_output(wait=True)
        print(f"Keeping alive... {time.strftime('%H:%M:%S')}")
        time.sleep(60)

# 在单独的cell中运行（不要阻塞训练）
# import threading
# t = threading.Thread(target=keep_alive)
# t.start()
```

### Q5: 使用Colab Pro获得更好的GPU
- Colab Free: T4 GPU (16GB)
- Colab Pro: V100 (16GB) 或 A100 (40GB)
- Colab Pro+: 更长的运行时间，后台执行

---

## 7. 完整Notebook代码

以下是可以直接复制到Colab的完整代码：

```python
#@title 1. 环境配置
!pip install timm einops pyyaml tqdm matplotlib seaborn scikit-learn kaggle --quiet
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

#@title 2. 配置Kaggle API
from google.colab import files
print("请上传kaggle.json文件:")
uploaded = files.upload()
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

#@title 3. 挂载Google Drive并下载数据
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = '/content/drive/MyDrive/HMS_Data'
!mkdir -p {DATA_DIR}

import os
if not os.path.exists(f'{DATA_DIR}/train.csv'):
    !kaggle competitions download -c hms-harmful-brain-activity-classification -p {DATA_DIR}
    !cd {DATA_DIR} && unzip -q '*.zip'
    print("✓ 数据下载完成!")
else:
    print("✓ 数据已存在")

#@title 4. 运行训练 {display-mode: "form"}
# 复制上面的train_colab.py内容到这里
# 然后运行 main()

#@title 5. 查看结果
from IPython.display import Image
Image('/content/drive/MyDrive/HMS_Output/training_curves.png')
```

---

## 提示

1. **首次运行**：数据下载可能需要30-60分钟，建议保存到Google Drive
2. **GPU选择**：`运行时` → `更改运行时类型` → 选择GPU
3. **长时间训练**：使用Colab Pro或定期保存检查点
4. **结果持久化**：始终保存到Google Drive，避免运行时断开丢失数据

---

**Happy Training! 🚀**
