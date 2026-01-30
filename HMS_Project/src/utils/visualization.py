"""
HMS训练可视化模块
自动生成并保存训练过程中的各种图表
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 服务器无显示器时使用
import seaborn as sns
from typing import Dict, List, Optional
import json
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """
    训练可视化器

    功能：
    - 实时记录训练指标
    - 自动生成各种图表
    - 保存到指定目录
    """

    def __init__(self, save_dir: str, experiment_name: str = "hms_experiment"):
        """
        Args:
            save_dir: 图表保存目录
            experiment_name: 实验名称
        """
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.figures_dir = os.path.join(save_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)

        # 指标记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_kl': [],
            'val_kl': [],
            'train_contrastive': [],
            'train_cross_modal': [],
            'train_soft_label': [],
            'learning_rate': [],
            'epoch': [],
            'fusion_weights': [],  # 记录融合权重变化
        }

        # 时间戳
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def update(self, epoch: int, metrics: Dict[str, float], lr: float = None):
        """
        更新指标

        Args:
            epoch: 当前epoch
            metrics: 指标字典
            lr: 当前学习率
        """
        self.history['epoch'].append(epoch)

        for key in ['train_loss', 'val_loss', 'train_kl', 'val_kl',
                    'train_contrastive', 'train_cross_modal', 'train_soft_label']:
            if key in metrics:
                self.history[key].append(metrics[key])
            elif key not in ['train_cross_modal', 'train_soft_label']:
                # 这些是可选的，不存在时填0
                self.history[key].append(0)

        if lr is not None:
            self.history['learning_rate'].append(lr)

        # 保存历史记录到JSON
        self._save_history()

    def update_fusion_weights(self, weights: np.ndarray):
        """记录融合权重"""
        self.history['fusion_weights'].append(weights.tolist())

    def _save_history(self):
        """保存历史记录到JSON文件"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_all(self):
        """生成所有图表"""
        self.plot_loss_curves()
        self.plot_kl_curves()
        self.plot_learning_rate()
        self.plot_contrastive_losses()
        if self.history['fusion_weights']:
            self.plot_fusion_weights()
        self.plot_summary()

    def plot_loss_curves(self):
        """绘制训练/验证损失曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history['epoch']
        train_loss = self.history['train_loss']
        val_loss = self.history['val_loss']

        ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)

        # 标记最佳验证损失
        if val_loss:
            best_epoch = epochs[np.argmin(val_loss)]
            best_loss = min(val_loss)
            ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
            ax.scatter([best_epoch], [best_loss], color='g', s=100, zorder=5)
            ax.annotate(f'Best: {best_loss:.4f}\nEpoch {best_epoch}',
                       xy=(best_epoch, best_loss),
                       xytext=(best_epoch + 1, best_loss + 0.1),
                       fontsize=10)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'loss_curves.png'), dpi=150)
        plt.close()

    def plot_kl_curves(self):
        """绘制KL散度曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history['epoch']
        train_kl = self.history['train_kl']
        val_kl = self.history['val_kl']

        ax.plot(epochs, train_kl, 'b-', label='Train KL', linewidth=2)
        ax.plot(epochs, val_kl, 'r-', label='Val KL', linewidth=2)

        # 标记最佳
        if val_kl:
            best_epoch = epochs[np.argmin(val_kl)]
            best_kl = min(val_kl)
            ax.scatter([best_epoch], [best_kl], color='g', s=100, zorder=5)
            ax.annotate(f'Best KL: {best_kl:.4f}',
                       xy=(best_epoch, best_kl),
                       xytext=(best_epoch + 1, best_kl + 0.05),
                       fontsize=10)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title('KL Divergence (Competition Metric)', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'kl_curves.png'), dpi=150)
        plt.close()

    def plot_learning_rate(self):
        """绘制学习率曲线"""
        if not self.history['learning_rate']:
            return

        fig, ax = plt.subplots(figsize=(10, 4))

        epochs = self.history['epoch']
        lr = self.history['learning_rate']

        ax.plot(epochs, lr, 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'learning_rate.png'), dpi=150)
        plt.close()

    def plot_contrastive_losses(self):
        """绘制各项对比学习损失"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs = self.history['epoch']

        # 监督对比损失
        if any(self.history['train_contrastive']):
            axes[0].plot(epochs, self.history['train_contrastive'], 'b-', linewidth=2)
            axes[0].set_title('Supervised Contrastive Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)

        # 跨模态对比损失
        if 'train_cross_modal' in self.history and any(self.history.get('train_cross_modal', [])):
            axes[1].plot(epochs, self.history['train_cross_modal'], 'r-', linewidth=2)
            axes[1].set_title('Cross-Modal Contrastive Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].grid(True, alpha=0.3)

        # 软标签对比损失
        if 'train_soft_label' in self.history and any(self.history.get('train_soft_label', [])):
            axes[2].plot(epochs, self.history['train_soft_label'], 'g-', linewidth=2)
            axes[2].set_title('Soft-Label Contrastive Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Loss')
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'contrastive_losses.png'), dpi=150)
        plt.close()

    def plot_fusion_weights(self):
        """绘制融合权重变化"""
        if not self.history['fusion_weights']:
            return

        weights = np.array(self.history['fusion_weights'])

        fig, ax = plt.subplots(figsize=(10, 6))

        modality_names = ['EEG', 'EEG-Spec', 'Spec'] if weights.shape[1] == 3 else ['EEG', 'Spec']

        for i, name in enumerate(modality_names):
            ax.plot(weights[:, i], label=name, linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title('Fusion Weights Over Training', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'fusion_weights.png'), dpi=150)
        plt.close()

    def plot_summary(self):
        """生成训练总结图"""
        fig = plt.figure(figsize=(16, 12))

        # 2x2布局
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        epochs = self.history['epoch']

        # 1. 损失曲线
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        ax1.set_title('Loss Curves', fontsize=12)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. KL散度
        ax2.plot(epochs, self.history['train_kl'], 'b-', label='Train', linewidth=2)
        ax2.plot(epochs, self.history['val_kl'], 'r-', label='Val', linewidth=2)
        ax2.set_title('KL Divergence', fontsize=12)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('KL')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 学习率
        if self.history['learning_rate']:
            ax3.plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
            ax3.set_yscale('log')
        ax3.set_title('Learning Rate', fontsize=12)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.grid(True, alpha=0.3)

        # 4. 最终统计
        stats_text = f"""
Training Summary
================
Experiment: {self.experiment_name}
Start Time: {self.start_time}

Best Val Loss: {min(self.history['val_loss']):.4f}
Best Val KL: {min(self.history['val_kl']):.4f}
Best Epoch: {epochs[np.argmin(self.history['val_kl'])]}

Final Train Loss: {self.history['train_loss'][-1]:.4f}
Final Val Loss: {self.history['val_loss'][-1]:.4f}
Final Val KL: {self.history['val_kl'][-1]:.4f}

Total Epochs: {len(epochs)}
"""
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax4.transAxes)
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'training_summary.png'), dpi=150)
        plt.close()

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        class_names: List[str] = None):
        """
        绘制预测结果分析

        Args:
            y_true: (N, 6) 真实标签
            y_pred: (N, 6) 预测概率
            class_names: 类别名称
        """
        if class_names is None:
            class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 每个类别的预测分布
        for i, (ax, name) in enumerate(zip(axes.flatten(), class_names)):
            ax.hist(y_pred[:, i], bins=50, alpha=0.7, label='Predicted', density=True)
            ax.hist(y_true[:, i], bins=50, alpha=0.7, label='True', density=True)
            ax.set_title(f'{name}')
            ax.set_xlabel('Probability')
            ax.set_ylabel('Density')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'prediction_distributions.png'), dpi=150)
        plt.close()

        # 混淆矩阵（使用argmax）
        fig, ax = plt.subplots(figsize=(10, 8))

        y_true_class = y_true.argmax(axis=1)
        y_pred_class = y_pred.argmax(axis=1)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_class, y_pred_class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Normalized Confusion Matrix', fontsize=14)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    def plot_sample_predictions(self, eeg_sample: np.ndarray, spec_sample: np.ndarray,
                                true_label: np.ndarray, pred_label: np.ndarray,
                                sample_idx: int = 0):
        """
        绘制单个样本的预测可视化

        Args:
            eeg_sample: (20, T) EEG信号
            spec_sample: (4, H, W) 频谱图
            true_label: (6,) 真实标签
            pred_label: (6,) 预测标签
            sample_idx: 样本索引
        """
        class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

        fig = plt.figure(figsize=(16, 10))

        # EEG信号（选择4个代表性通道）
        ax1 = fig.add_subplot(2, 2, 1)
        channels_to_plot = [0, 5, 10, 15]  # Fp1, C3, P3, O1
        for i, ch in enumerate(channels_to_plot):
            ax1.plot(eeg_sample[ch] + i * 2, label=f'Ch{ch}')
        ax1.set_title('EEG Signal (4 channels)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.legend(loc='upper right')

        # 频谱图（4个脑区平均）
        ax2 = fig.add_subplot(2, 2, 2)
        spec_avg = spec_sample.mean(axis=0)
        im = ax2.imshow(spec_avg, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title('Spectrogram (4-region average)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Frequency')
        plt.colorbar(im, ax=ax2)

        # 预测vs真实
        ax3 = fig.add_subplot(2, 2, 3)
        x = np.arange(len(class_names))
        width = 0.35
        ax3.bar(x - width/2, true_label, width, label='True', color='blue', alpha=0.7)
        ax3.bar(x + width/2, pred_label, width, label='Predicted', color='red', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(class_names, rotation=45)
        ax3.set_ylabel('Probability')
        ax3.set_title('True vs Predicted Distribution')
        ax3.legend()

        # 预测信息
        ax4 = fig.add_subplot(2, 2, 4)
        true_class = class_names[true_label.argmax()]
        pred_class = class_names[pred_label.argmax()]
        kl_div = np.sum(true_label * np.log(true_label / (pred_label + 1e-8) + 1e-8))

        info_text = f"""
Sample #{sample_idx}

True Class: {true_class}
Predicted Class: {pred_class}
Correct: {'Yes' if true_class == pred_class else 'No'}

KL Divergence: {kl_div:.4f}

True Distribution:
{', '.join([f'{name}: {val:.2f}' for name, val in zip(class_names, true_label)])}

Predicted Distribution:
{', '.join([f'{name}: {val:.2f}' for name, val in zip(class_names, pred_label)])}
"""
        ax4.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax4.transAxes)
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f'sample_prediction_{sample_idx}.png'), dpi=150)
        plt.close()


def plot_training_curves_from_history(history_path: str, output_dir: str):
    """
    从保存的历史记录重新生成图表

    Args:
        history_path: training_history.json的路径
        output_dir: 输出目录
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    visualizer = TrainingVisualizer(output_dir, "restored")
    visualizer.history = history
    visualizer.plot_all()
    print(f"Charts saved to {output_dir}/figures/")


if __name__ == "__main__":
    # 测试代码
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, help='Path to training_history.json')
    parser.add_argument('--output', type=str, default='./outputs', help='Output directory')
    args = parser.parse_args()

    if args.history:
        plot_training_curves_from_history(args.history, args.output)
