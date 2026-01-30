# HMS数据集关键问题分析与解决方案

## 一、数据集核心发现

### 1.1 数据规模概览

| 项目 | 数量 |
|------|------|
| 总样本数 | 106,800 |
| 唯一EEG文件 | 17,089 |
| 唯一Spectrogram文件 | 11,138 |
| 患者数 | 1,950 |

---

## 二、问题一：患者数据量严重不平衡

### 2.1 问题描述

```
患者样本数统计：
  最小: 1 个样本
  最大: 2,215 个样本
  中位数: 19 个样本
  平均值: 54.8 个样本
  标准差: 133.2

前5名患者样本数：
  患者30631: 2,215样本 (占比2.1%)
  患者2641:  2,185样本 (占比2.0%)
  患者35627: 1,403样本 (占比1.3%)
  患者28330: 1,362样本 (占比1.3%)
  患者54199: 1,350样本 (占比1.3%)

后5名患者：仅1个样本
```

### 2.2 影响

- 模型可能过拟合到样本量大的患者的特征
- 验证集划分不当会导致数据泄露
- 少样本患者的模式难以学习

### 2.3 解决方案

#### 方案A：GroupKFold交叉验证（必须）

```python
from sklearn.model_selection import GroupKFold

# 按patient_id分组，确保同一患者不同时出现在训练和验证集
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['patient_id'])):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
```

#### 方案B：患者级别重采样

```python
def patient_balanced_sampler(df, samples_per_patient=50):
    """
    对每个患者进行重采样，平衡数据量
    """
    balanced_dfs = []
    for patient_id, group in df.groupby('patient_id'):
        if len(group) > samples_per_patient:
            # 下采样
            sampled = group.sample(n=samples_per_patient, random_state=42)
        else:
            # 上采样（有放回）
            sampled = group.sample(n=samples_per_patient, replace=True, random_state=42)
        balanced_dfs.append(sampled)
    return pd.concat(balanced_dfs, ignore_index=True)
```

#### 方案C：加权损失函数

```python
def compute_patient_weights(df):
    """
    计算每个样本的权重，样本量少的患者权重高
    """
    patient_counts = df.groupby('patient_id').size()
    max_count = patient_counts.max()

    # 权重 = sqrt(max_count / patient_count)
    weights = df['patient_id'].map(lambda x: np.sqrt(max_count / patient_counts[x]))
    return weights

# 在DataLoader中使用WeightedRandomSampler
from torch.utils.data import WeightedRandomSampler

weights = compute_patient_weights(train_df)
sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True)
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

---

## 三、问题二：EEG和Spectrogram对接复杂

### 3.1 问题描述

**关键发现：EEG和Spectrogram是独立的数据源！**

```
1. 数量不匹配：
   - EEG文件: 17,089个
   - Spectrogram文件: 11,138个

2. 映射关系复杂：
   - 1个Spectrogram可对应1-21个EEG（同一患者不同时段）
   - 8,758个Spectrogram只对应1个EEG
   - 2,380个Spectrogram对应多个EEG

3. Offset不同步：
   - 29,561个样本(28%) EEG和Spectrogram的offset不相等！
   - offset差异范围: -17,556秒 到 -50秒
```

### 3.2 数据结构图解

```
train.csv 每行代表一个标注样本:
┌──────────────────────────────────────────────────────────────────┐
│ label_id | eeg_id | eeg_offset | spec_id | spec_offset | votes  │
├──────────────────────────────────────────────────────────────────┤
│ 样本1    │ EEG_A  │ 0秒        │ Spec_X  │ 0秒         │ [3,0..]│
│ 样本2    │ EEG_A  │ 6秒        │ Spec_X  │ 6秒         │ [3,0..]│
│ 样本3    │ EEG_B  │ 0秒        │ Spec_X  │ 74秒        │ [2,1..]│ ← offset不同！
│ 样本4    │ EEG_C  │ 0秒        │ Spec_X  │ 190秒       │ [0,3..]│ ← 另一个EEG共享Spec
└──────────────────────────────────────────────────────────────────┘
          ↓                              ↓
    ┌─────────────┐              ┌─────────────────┐
    │ EEG文件     │              │ Spectrogram文件  │
    │ 200Hz采样   │              │ 每2秒一帧       │
    │ 50-90秒长   │              │ 数百秒长        │
    └─────────────┘              └─────────────────┘
```

### 3.3 正确的数据提取逻辑

```python
def extract_eeg_window(eeg_path, offset_seconds, window_seconds=10, fs=200):
    """
    从EEG文件中提取指定时间窗口

    Args:
        eeg_path: EEG parquet文件路径
        offset_seconds: 起始时间偏移（秒）
        window_seconds: 窗口长度（秒），默认10秒
        fs: 采样率，默认200Hz

    Returns:
        numpy array, shape (window_seconds * fs, n_channels)
    """
    eeg_df = pd.read_parquet(eeg_path)
    eeg_data = eeg_df.values  # (total_samples, 20)

    start_idx = int(offset_seconds * fs)
    end_idx = start_idx + int(window_seconds * fs)

    # 边界检查
    if end_idx > len(eeg_data):
        # 如果超出范围，取最后10秒
        end_idx = len(eeg_data)
        start_idx = max(0, end_idx - int(window_seconds * fs))

    return eeg_data[start_idx:end_idx]


def extract_spectrogram_window(spec_path, offset_seconds, window_seconds=10):
    """
    从Spectrogram文件中提取指定时间窗口

    Args:
        spec_path: Spectrogram parquet文件路径
        offset_seconds: 中心时间偏移（秒）
        window_seconds: 窗口长度（秒）

    Returns:
        numpy array, shape (n_time_steps, n_freq_bins)
    """
    spec_df = pd.read_parquet(spec_path)

    # time列是秒数
    time_col = spec_df['time'].values

    # 找到offset附近的窗口
    # 假设标注点在窗口中心
    center_time = offset_seconds + window_seconds / 2

    # 找最近的5帧（10秒，每帧2秒）
    mask = (time_col >= offset_seconds) & (time_col <= offset_seconds + window_seconds)

    if mask.sum() == 0:
        # 如果找不到，取最近的帧
        closest_idx = np.abs(time_col - center_time).argmin()
        start_idx = max(0, closest_idx - 2)
        end_idx = min(len(time_col), closest_idx + 3)
        mask = np.zeros(len(time_col), dtype=bool)
        mask[start_idx:end_idx] = True

    # 提取数据（排除time列）
    feature_cols = [c for c in spec_df.columns if c != 'time']
    spec_data = spec_df.loc[mask, feature_cols].values

    return spec_data
```

### 3.4 完整的Dataset实现

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class HMSDataset(Dataset):
    def __init__(self, df, eeg_dir, spec_dir, mode='train'):
        self.df = df.reset_index(drop=True)
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.mode = mode

        self.target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote',
                           'lrda_vote', 'grda_vote', 'other_vote']

        # 缓存已加载的文件（节省IO）
        self.eeg_cache = {}
        self.spec_cache = {}

    def __len__(self):
        return len(self.df)

    def _load_eeg(self, eeg_id):
        if eeg_id not in self.eeg_cache:
            path = f"{self.eeg_dir}/{eeg_id}.parquet"
            self.eeg_cache[eeg_id] = pd.read_parquet(path).values
        return self.eeg_cache[eeg_id]

    def _load_spec(self, spec_id):
        if spec_id not in self.spec_cache:
            path = f"{self.spec_dir}/{spec_id}.parquet"
            self.spec_cache[spec_id] = pd.read_parquet(path)
        return self.spec_cache[spec_id]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ============ 提取EEG ============
        eeg_id = int(row['eeg_id'])
        eeg_offset = row['eeg_label_offset_seconds']

        eeg_data = self._load_eeg(eeg_id)

        # 提取10秒窗口 (2000个点 @ 200Hz)
        start = int(eeg_offset * 200)
        end = start + 2000

        if end > len(eeg_data):
            end = len(eeg_data)
            start = max(0, end - 2000)

        eeg_window = eeg_data[start:end]  # (2000, 20)

        # 填充不足的情况
        if len(eeg_window) < 2000:
            pad_len = 2000 - len(eeg_window)
            eeg_window = np.pad(eeg_window, ((0, pad_len), (0, 0)), mode='constant')

        # ============ 提取Spectrogram ============
        spec_id = int(row['spectrogram_id'])
        spec_offset = row['spectrogram_label_offset_seconds']

        spec_df = self._load_spec(spec_id)
        time_col = spec_df['time'].values

        # 提取offset附近10秒的帧 (约5帧，每帧2秒)
        mask = (time_col >= spec_offset) & (time_col < spec_offset + 10)

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
            region_data = spec_df.loc[mask, cols].values  # (n_frames, 100)
            spec_images.append(region_data)

        # Stack成 (4, n_frames, 100)
        spec_window = np.stack(spec_images, axis=0)

        # 归一化并调整大小
        spec_window = np.nan_to_num(spec_window, nan=0.0)
        spec_window = np.log1p(np.clip(spec_window, 0, None))

        # Resize到固定大小 (4, 128, 256) 使用插值
        from scipy.ndimage import zoom
        spec_resized = np.zeros((4, 128, 256))
        for i in range(4):
            if spec_window[i].size > 0:
                h, w = spec_window[i].shape
                spec_resized[i] = zoom(spec_window[i], (128/h, 256/w), order=1)

        # ============ 标签 ============
        votes = row[self.target_cols].values.astype(np.float32)
        label = votes / (votes.sum() + 1e-8)  # 归一化为概率

        return {
            'eeg': torch.tensor(eeg_window.T, dtype=torch.float32),  # (20, 2000)
            'spec': torch.tensor(spec_resized, dtype=torch.float32),  # (4, 128, 256)
            'label': torch.tensor(label, dtype=torch.float32),  # (6,)
            'eeg_id': eeg_id,
            'spec_id': spec_id
        }
```

---

## 四、问题三：多个EEG共享同一Spectrogram

### 4.1 问题描述

```python
# 每个Spectrogram对应的EEG数量分布
1个EEG:  8,758个Spectrogram
2个EEG:  1,334个Spectrogram
3个EEG:    446个Spectrogram
...
21个EEG:    5个Spectrogram
```

### 4.2 影响

- 同一Spectrogram被多次使用，可能导致信息泄露
- 在训练/验证划分时需要考虑Spectrogram级别的分组

### 4.3 解决方案

```python
def create_leak_free_split(df, n_splits=5):
    """
    创建无泄露的数据划分
    同时考虑patient_id和spectrogram_id
    """
    from sklearn.model_selection import GroupKFold

    # 创建复合分组键
    df['group_key'] = df['patient_id'].astype(str) + '_' + df['spectrogram_id'].astype(str)

    # 获取唯一的分组
    unique_groups = df.groupby('group_key').first().reset_index()

    # 对唯一分组进行KFold
    gkf = GroupKFold(n_splits=n_splits)

    folds = []
    for train_groups, val_groups in gkf.split(unique_groups, groups=unique_groups['patient_id']):
        train_keys = unique_groups.iloc[train_groups]['group_key'].values
        val_keys = unique_groups.iloc[val_groups]['group_key'].values

        train_idx = df[df['group_key'].isin(train_keys)].index.tolist()
        val_idx = df[df['group_key'].isin(val_keys)].index.tolist()

        folds.append((train_idx, val_idx))

    return folds
```

---

## 五、问题四：投票数据不均匀

### 5.1 问题描述

```
每行总投票数分布:
  1票:  4,360样本
  2票:  2,316样本
  3票: 51,867样本 (最多)
  ...
  22票:   54样本
```

### 5.2 解决方案

投票数已经是软标签形式，只需归一化：

```python
def create_soft_labels(row):
    """
    将投票数转换为概率分布
    """
    votes = np.array([
        row['seizure_vote'],
        row['lpd_vote'],
        row['gpd_vote'],
        row['lrda_vote'],
        row['grda_vote'],
        row['other_vote']
    ], dtype=np.float32)

    # 归一化
    total = votes.sum()
    if total > 0:
        probs = votes / total
    else:
        probs = np.ones(6) / 6  # 均匀分布

    return probs
```

---

## 六、完整的训练Pipeline

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

def main():
    # 1. 加载数据
    df = pd.read_csv('data/train.csv')

    # 2. 创建无泄露的划分
    folds = create_leak_free_split(df, n_splits=5)

    # 3. 训练每个fold
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/5")
        print(f"{'='*50}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        print(f"训练集: {len(train_df)} 样本, {train_df['patient_id'].nunique()} 患者")
        print(f"验证集: {len(val_df)} 样本, {val_df['patient_id'].nunique()} 患者")

        # 4. 创建Dataset
        train_dataset = HMSDataset(train_df, 'data/train_eegs', 'data/train_spectrograms', mode='train')
        val_dataset = HMSDataset(val_df, 'data/train_eegs', 'data/train_spectrograms', mode='val')

        # 5. 患者平衡采样
        patient_weights = compute_patient_weights(train_df)
        sampler = WeightedRandomSampler(patient_weights.values, len(train_df), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        # 6. 训练模型
        model = HMSModel().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        best_val_loss = float('inf')
        for epoch in range(20):
            # 训练
            model.train()
            train_loss = 0
            for batch in train_loader:
                eeg = batch['eeg'].cuda()
                spec = batch['spec'].cuda()
                label = batch['label'].cuda()

                pred = model(eeg, spec)
                loss = F.kl_div(pred.log(), label, reduction='batchmean')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].cuda()
                    spec = batch['spec'].cuda()
                    label = batch['label'].cuda()

                    pred = model(eeg, spec)
                    loss = F.kl_div(pred.log(), label, reduction='batchmean')
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'models/fold{fold}_best.pth')

if __name__ == '__main__':
    main()
```

---

## 七、总结：关键注意事项

| 问题 | 解决方案 | 重要程度 |
|------|----------|----------|
| 患者数据不平衡 | GroupKFold + 加权采样 | ⭐⭐⭐⭐⭐ |
| EEG-Spec offset不同 | 分别按各自offset提取 | ⭐⭐⭐⭐⭐ |
| 多EEG共享Spec | 复合分组键划分 | ⭐⭐⭐⭐ |
| 投票数不均 | 归一化为概率分布 | ⭐⭐⭐ |
| EEG长度不一 | 固定10秒窗口+padding | ⭐⭐⭐ |
| Spec帧数不一 | Resize到固定大小 | ⭐⭐⭐ |
