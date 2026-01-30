# HMS 服务器运行快速指南

## 1. 环境配置

### 1.1 创建虚拟环境
```bash
# 使用conda
conda create -n hms python=3.10
conda activate hms

# 或者使用venv
python -m venv hms_env
source hms_env/bin/activate
```

### 1.2 安装依赖
```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 项目依赖
pip install -r requirements.txt

# 安装Mamba（可选，需要CUDA）
pip install mamba-ssm --no-build-isolation
```

## 2. 数据下载

### 2.1 配置Kaggle API
```bash
# 1. 访问 https://www.kaggle.com/settings
# 2. 点击 "Create New Token" 下载 kaggle.json
# 3. 配置到服务器
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2.2 下载数据
```bash
# 下载比赛数据（约50GB）
python scripts/download_data.py --data_dir ./data

# 或者使用Kaggle命令行
kaggle competitions download -c hms-harmful-brain-activity-classification -p ./data
```

### 2.3 更新配置文件
```bash
# 编辑 configs/config.yaml，修改数据路径
vim configs/config.yaml

# 修改以下内容：
# data:
#   train_csv: "/path/to/your/data/train.csv"
#   eeg_dir: "/path/to/your/data/train_eegs"
#   spec_dir: "/path/to/your/data/train_spectrograms"
```

## 3. 开始训练

### 3.1 前台运行（调试用）
```bash
python scripts/train.py --config configs/config.yaml
```

### 3.2 后台运行（推荐）
```bash
# 使用nohup后台运行
nohup python scripts/train.py --config configs/config.yaml > train.log 2>&1 &

# 查看进程ID
echo $!

# 查看训练日志
tail -f train.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

### 3.3 使用tmux/screen（推荐）
```bash
# 创建新会话
tmux new -s hms_train

# 在tmux中运行
python scripts/train.py --config configs/config.yaml

# 分离会话：Ctrl+B, D
# 重新连接：tmux attach -t hms_train
```

### 3.4 Debug模式（小数据集测试）
```bash
python scripts/train.py --config configs/config.yaml --debug
```

## 4. 监控训练

### 4.1 查看训练日志
```bash
tail -f train.log
```

### 4.2 查看GPU使用
```bash
nvidia-smi
# 或
watch -n 1 nvidia-smi
```

### 4.3 查看训练曲线
训练完成后，图表会自动保存到：
```
outputs/hms_experiment_YYYYMMDD_HHMMSS/figures/
├── loss_curves.png           # 损失曲线
├── kl_curves.png             # KL散度曲线
├── learning_rate.png         # 学习率曲线
├── contrastive_losses.png    # 对比学习损失
├── fusion_weights.png        # 融合权重变化
├── training_summary.png      # 训练总结
├── prediction_distributions.png  # 预测分布
└── confusion_matrix.png      # 混淆矩阵
```

### 4.4 从历史记录恢复图表
```bash
python src/utils/visualization.py \
    --history outputs/xxx/training_history.json \
    --output ./restored_figures
```

## 5. 输出文件说明

训练完成后，输出目录结构：
```
outputs/hms_experiment_YYYYMMDD_HHMMSS/
├── config.yaml               # 实验配置备份
├── results.json              # 最终结果
├── training_history.json     # 训练历史数据
├── checkpoints/
│   ├── best_model.pth        # 最佳模型
│   └── checkpoint_epoch*.pth # 定期检查点
├── figures/                  # 所有可视化图表
│   └── *.png
└── logs/
    └── *.log                 # 训练日志
```

## 6. 常用命令

```bash
# 查看所有进程
ps aux | grep train.py

# 结束训练
kill -9 <PID>

# 清理缓存
rm -rf __pycache__ .cache

# 查看磁盘使用
df -h

# 查看目录大小
du -sh data/
du -sh outputs/

# 打包实验结果
tar -czvf results.tar.gz outputs/hms_experiment_*/
```

## 7. 常见问题

### Q: CUDA内存不足
```bash
# 减小batch_size
vim configs/config.yaml
# 修改 training.batch_size: 16

# 或增加梯度累积
# 修改 training.gradient_accumulation_steps: 4
```

### Q: 数据加载慢
```bash
# 增加num_workers
vim configs/config.yaml
# 修改 training.num_workers: 8
```

### Q: 训练中断如何恢复
```bash
python scripts/train.py --config configs/config.yaml \
    --resume outputs/xxx/checkpoints/checkpoint_epoch9.pth
```

### Q: Mamba安装失败
```bash
# 使用备用方案（自动fallback到BiLSTM）
# 或安装causal-conv1d
pip install causal-conv1d>=1.1.0
```

## 8. 多GPU训练（可选）

```bash
# 使用DataParallel
# 在train.py中取消注释相关代码

# 或使用torchrun
torchrun --nproc_per_node=4 scripts/train.py --config configs/config.yaml
```

## 9. 推理和提交

```bash
# 生成提交文件
python scripts/inference.py \
    --checkpoint outputs/xxx/checkpoints/best_model.pth \
    --test_dir ./data/test_eegs \
    --output submission.csv
```

---

**祝训练顺利！** 🚀
