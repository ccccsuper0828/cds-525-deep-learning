"""
KMNIST (Kuzushiji-MNIST) Dataset - Exploratory Data Analysis
数据集: https://github.com/rois-codh/kmnist
PyTorch内置: torchvision.datasets.KMNIST
输出: HTML报告
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import base64
from io import BytesIO, StringIO
from datetime import datetime

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = '../reports/figures/kmnist'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class HTMLReport:
    """HTML报告生成器"""
    def __init__(self, title):
        self.title = title
        self.sections = []
        
    def add_section(self, title, content):
        self.sections.append({'title': title, 'content': content, 'type': 'text'})
    
    def add_figure(self, fig, title=""):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        self.sections.append({'title': title, 'content': img_base64, 'type': 'image'})
    
    def generate_html(self):
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Times New Roman', Georgia, serif;
            background: #ffffff;
            min-height: 100vh;
            color: #1a1a1a;
            line-height: 1.8;
            font-size: 11pt;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 50px;
        }}
        h1 {{
            font-size: 1.8em;
            font-weight: normal;
            color: #1a1a1a;
            text-align: center;
            margin-bottom: 8px;
            border-bottom: none;
        }}
        .meta {{
            text-align: center;
            color: #555;
            font-size: 0.95em;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ccc;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            font-size: 1.3em;
            font-weight: bold;
            color: #1a1a1a;
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 1px solid #333;
        }}
        .section pre {{
            background: #f9f9f9;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Courier New', Consolas, monospace;
            font-size: 9pt;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #1a1a1a;
            border: 1px solid #ddd;
            margin: 15px 0;
        }}
        .section img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
        }}
        .figure-caption {{
            text-align: center;
            font-size: 0.9em;
            color: #555;
            margin-top: 8px;
            font-style: italic;
        }}
        footer {{
            text-align: center;
            padding: 30px 0;
            color: #777;
            font-size: 0.85em;
            border-top: 1px solid #ccc;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.title}</h1>
        <p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
'''
        for section in self.sections:
            html += f'''
        <div class="section">
            <h2>{section['title']}</h2>
'''
            if section['type'] == 'text':
                html += f'            <pre>{section["content"]}</pre>\n'
            else:
                html += f'            <img src="data:image/png;base64,{section["content"]}" alt="{section["title"]}">\n'
            html += '        </div>\n'
        
        html += '''
        <footer>
            <p>KMNIST Dataset - EDA Report</p>
        </footer>
    </div>
</body>
</html>'''
        return html


# 全局报告对象
report = HTMLReport("KMNIST (Kuzushiji-MNIST) Dataset EDA")


# KMNIST类别映射 (日语平假名)
CLASS_NAMES = {
    0: 'お (o)',
    1: 'き (ki)',
    2: 'す (su)',
    3: 'つ (tsu)',
    4: 'な (na)',
    5: 'は (ha)',
    6: 'ま (ma)',
    7: 'や (ya)',
    8: 'れ (re)',
    9: 'を (wo)'
}


def load_data(data_dir='../data'):
    """加载KMNIST数据集"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("1. 数据加载\n")
    output.write("=" * 60 + "\n")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    try:
        train_dataset = torchvision.datasets.KMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.KMNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        output.write("数据集下载/加载成功!\n")
    except Exception as e:
        output.write(f"[ERROR] 下载失败: {e}\n")
        output.write("请检查网络连接或手动下载数据集\n")
        report.add_section("1. 数据加载", output.getvalue())
        raise RuntimeError(f"无法加载KMNIST数据集: {e}")

    output.write(f"\n训练集大小: {len(train_dataset)}\n")
    output.write(f"测试集大小: {len(test_dataset)}\n")
    output.write(f"图像尺寸: {train_dataset[0][0].shape}\n")
    output.write(f"类别数量: {len(CLASS_NAMES)}\n")

    report.add_section("1. 数据加载", output.getvalue())
    return train_dataset, test_dataset


def basic_info(train_dataset, test_dataset):
    """基本信息统计"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("2. 基本信息统计\n")
    output.write("=" * 60 + "\n")

    # 获取标签
    if hasattr(train_dataset, 'targets'):
        train_labels = train_dataset.targets
        test_labels = test_dataset.targets
    else:
        train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
        test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.numpy()
        test_labels = test_labels.numpy()

    output.write(f"\n训练集:\n")
    output.write(f"  样本数: {len(train_dataset)}\n")
    output.write(f"  每类样本数: {len(train_dataset) // 10}\n")

    output.write(f"\n测试集:\n")
    output.write(f"  样本数: {len(test_dataset)}\n")
    output.write(f"  每类样本数: {len(test_dataset) // 10}\n")

    output.write(f"\n图像信息:\n")
    sample_image, _ = train_dataset[0]
    output.write(f"  形状: {sample_image.shape}\n")
    output.write(f"  数据类型: {sample_image.dtype}\n")
    output.write(f"  像素值范围: [{sample_image.min():.3f}, {sample_image.max():.3f}]\n")

    report.add_section("2. 基本信息统计", output.getvalue())
    return train_labels, test_labels


def class_distribution(train_labels, test_labels):
    """类别分布分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("3. 类别分布分析\n")
    output.write("=" * 60 + "\n")

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    output.write("\n训练集类别分布:\n")
    for cls in sorted(train_counts.keys()):
        output.write(f"  {CLASS_NAMES[cls]}: {train_counts[cls]}\n")

    output.write("\n测试集类别分布:\n")
    for cls in sorted(test_counts.keys()):
        output.write(f"  {CLASS_NAMES[cls]}: {test_counts[cls]}\n")

    # 验证平衡性
    classes = sorted(train_counts.keys())
    train_values = [train_counts[c] for c in classes]
    balance_ratio = min(train_values) / max(train_values)
    output.write(f"\n类别平衡比: {balance_ratio:.3f} (1.0 = 完美平衡)\n")

    report.add_section("3. 类别分布分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 训练集分布
    test_values = [test_counts[c] for c in classes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))

    bars1 = axes[0].bar(classes, train_values, color=colors)
    axes[0].set_xlabel('Class')
    # 使用罗马字标签以避免日文字符显示问题
    roman_labels = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
    
    axes[0].set_ylabel('Count')
    axes[0].set_title('Training Set Class Distribution')
    axes[0].set_xticks(classes)
    axes[0].set_xticklabels([roman_labels[c] for c in classes])

    # 测试集分布
    bars2 = axes[1].bar(classes, test_values, color=colors)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Test Set Class Distribution')
    axes[1].set_xticks(classes)
    axes[1].set_xticklabels([roman_labels[c] for c in classes])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/class_distribution.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "类别分布可视化")


def sample_visualization(train_dataset):
    """样本可视化"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("4. 样本可视化\n")
    output.write("=" * 60 + "\n")
    output.write("展示每个类别的代表性样本图像\n")

    report.add_section("4. 样本可视化", output.getvalue())

    # 每个类别显示5个样本
    fig, axes = plt.subplots(10, 5, figsize=(12, 20))

    # 获取每个类别的样本索引
    if hasattr(train_dataset, 'targets'):
        targets = train_dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
    else:
        targets = np.array([train_dataset[i][1] for i in range(min(10000, len(train_dataset)))])

    for class_idx in range(10):
        # 找到该类别的样本
        class_indices = np.where(targets == class_idx)[0][:5]

        for j, idx in enumerate(class_indices):
            image, label = train_dataset[idx]
            if isinstance(image, torch.Tensor):
                image = image.numpy()

            ax = axes[class_idx, j]
            ax.imshow(image.squeeze(), cmap='gray')
            ax.axis('off')

            if j == 0:
                ax.set_ylabel(CLASS_NAMES[class_idx], fontsize=10)
                ax.yaxis.set_visible(True)
                ax.set_yticks([])

    fig.suptitle('KMNIST Sample Images (5 samples per class)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sample_images.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "样本图像展示")


def pixel_statistics(train_dataset):
    """像素统计分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("5. 像素统计分析\n")
    output.write("=" * 60 + "\n")

    # 采样分析 (避免内存问题)
    n_samples = min(10000, len(train_dataset))

    pixel_values = []
    for i in range(n_samples):
        image, _ = train_dataset[i]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        pixel_values.extend(image.flatten())

    pixel_values = np.array(pixel_values)

    output.write(f"\n像素值统计 (基于{n_samples}个样本):\n")
    output.write(f"  均值: {pixel_values.mean():.4f}\n")
    output.write(f"  标准差: {pixel_values.std():.4f}\n")
    output.write(f"  最小值: {pixel_values.min():.4f}\n")
    output.write(f"  最大值: {pixel_values.max():.4f}\n")
    output.write(f"  中位数: {np.median(pixel_values):.4f}\n")

    report.add_section("5. 像素统计分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 像素值直方图
    axes[0].hist(pixel_values, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0].axvline(pixel_values.mean(), color='red', linestyle='--',
                   label=f'Mean: {pixel_values.mean():.3f}')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Pixel Value Distribution')
    axes[0].legend()

    # 平均图像
    avg_images = {}
    if hasattr(train_dataset, 'targets'):
        targets = train_dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
    else:
        targets = np.array([train_dataset[i][1] for i in range(n_samples)])

    for class_idx in range(10):
        class_indices = np.where(targets[:n_samples] == class_idx)[0][:500]
        class_images = []
        for idx in class_indices:
            image, _ = train_dataset[idx]
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            class_images.append(image.squeeze())
        avg_images[class_idx] = np.mean(class_images, axis=0)

    # 显示所有类别的平均图像
    avg_all = np.stack(list(avg_images.values()))
    axes[1].imshow(avg_all.reshape(2, 5, 28, 28).transpose(0, 2, 1, 3).reshape(56, 140), cmap='gray')
    axes[1].set_title('Average Image per Class')
    axes[1].axis('off')

    # 像素强度热力图 (整体平均)
    overall_avg = np.mean(list(avg_images.values()), axis=0)
    im = axes[2].imshow(overall_avg, cmap='hot')
    axes[2].set_title('Overall Average Pixel Intensity')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pixel_statistics.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "像素统计分析")


def class_similarity_analysis(train_dataset):
    """类别相似度分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("6. 类别相似度分析\n")
    output.write("=" * 60 + "\n")

    # 计算每个类别的平均图像
    n_samples = min(10000, len(train_dataset))

    if hasattr(train_dataset, 'targets'):
        targets = train_dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
    else:
        targets = np.array([train_dataset[i][1] for i in range(n_samples)])

    avg_images = {}
    for class_idx in range(10):
        class_indices = np.where(targets[:n_samples] == class_idx)[0][:500]
        class_images = []
        for idx in class_indices:
            image, _ = train_dataset[idx]
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            class_images.append(image.flatten())

        avg_images[class_idx] = np.mean(class_images, axis=0)

    # 计算类别间相似度 (余弦相似度)
    similarity_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            dot = np.dot(avg_images[i], avg_images[j])
            norm_i = np.linalg.norm(avg_images[i])
            norm_j = np.linalg.norm(avg_images[j])
            similarity_matrix[i, j] = dot / (norm_i * norm_j)

    output.write("\n类别相似度矩阵 (余弦相似度):\n")

    # 找出最相似的类别对
    similar_pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            similar_pairs.append((i, j, similarity_matrix[i, j]))

    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    output.write("\n最相似的类别对:\n")
    for i, j, sim in similar_pairs[:5]:
        output.write(f"  {CLASS_NAMES[i]} vs {CLASS_NAMES[j]}: {sim:.4f}\n")

    report.add_section("6. 类别相似度分析", output.getvalue())

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用罗马字标签以避免日文字符显示问题
    roman_labels = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']

    im = ax.imshow(similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(roman_labels)
    ax.set_yticklabels(roman_labels)
    ax.set_title('Class Similarity Matrix (Cosine Similarity)')
    plt.colorbar(im)

    # 添加数值标签
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/class_similarity.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "类别相似度分析")


def stroke_complexity_analysis(train_dataset):
    """笔画复杂度分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("7. 笔画复杂度分析\n")
    output.write("=" * 60 + "\n")

    n_samples = min(10000, len(train_dataset))

    if hasattr(train_dataset, 'targets'):
        targets = train_dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
    else:
        targets = np.array([train_dataset[i][1] for i in range(n_samples)])

    # 每个类别的复杂度指标
    class_complexity = {i: {'pixel_density': [], 'edge_density': [], 'variance': []} for i in range(10)}

    for i in range(n_samples):
        image, label = train_dataset[i]
        if isinstance(image, torch.Tensor):
            image = image.numpy().squeeze()
        else:
            image = np.array(image).squeeze()

        # 1. 像素密度 (非零像素比例) - 越高表示笔画越粗/多
        pixel_density = (image > 0.1).sum() / image.size

        # 2. 边缘密度 (使用简单的梯度近似)
        grad_x = np.abs(np.diff(image, axis=1))
        grad_y = np.abs(np.diff(image, axis=0))
        edge_density = (grad_x.sum() + grad_y.sum()) / image.size

        # 3. 像素值方差 - 越高表示笔画变化越复杂
        variance = image.var()

        class_complexity[label]['pixel_density'].append(pixel_density)
        class_complexity[label]['edge_density'].append(edge_density)
        class_complexity[label]['variance'].append(variance)

    # 计算每个类别的平均复杂度
    complexity_summary = {}
    for cls in range(10):
        complexity_summary[cls] = {
            'pixel_density': np.mean(class_complexity[cls]['pixel_density']),
            'edge_density': np.mean(class_complexity[cls]['edge_density']),
            'variance': np.mean(class_complexity[cls]['variance'])
        }

    # 综合复杂度得分 (归一化后求和)
    all_pixel = [complexity_summary[c]['pixel_density'] for c in range(10)]
    all_edge = [complexity_summary[c]['edge_density'] for c in range(10)]
    all_var = [complexity_summary[c]['variance'] for c in range(10)]

    for cls in range(10):
        norm_pixel = (complexity_summary[cls]['pixel_density'] - min(all_pixel)) / (max(all_pixel) - min(all_pixel) + 1e-8)
        norm_edge = (complexity_summary[cls]['edge_density'] - min(all_edge)) / (max(all_edge) - min(all_edge) + 1e-8)
        norm_var = (complexity_summary[cls]['variance'] - min(all_var)) / (max(all_var) - min(all_var) + 1e-8)
        complexity_summary[cls]['overall'] = (norm_pixel + norm_edge + norm_var) / 3

    output.write("\n各字符复杂度分析:\n")
    output.write("-" * 60 + "\n")
    output.write(f"{'字符':<12} {'像素密度':<12} {'边缘密度':<12} {'综合得分':<12}\n")
    output.write("-" * 60 + "\n")

    # 按综合得分排序
    sorted_classes = sorted(range(10), key=lambda x: complexity_summary[x]['overall'], reverse=True)

    for cls in sorted_classes:
        char_name = CLASS_NAMES[cls]
        output.write(f"{char_name:<12} {complexity_summary[cls]['pixel_density']:.4f}       "
                    f"{complexity_summary[cls]['edge_density']:.4f}       "
                    f"{complexity_summary[cls]['overall']:.4f}\n")

    output.write("\n说明:\n")
    output.write("  - 像素密度: 非零像素占比，反映笔画粗细/数量\n")
    output.write("  - 边缘密度: 像素梯度，反映笔画转折复杂度\n")
    output.write("  - 综合得分: 归一化后的综合复杂度 (0-1)\n")

    report.add_section("7. 笔画复杂度分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    roman_labels = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
    colors = plt.cm.Set3(np.linspace(0, 1, 10))

    # 像素密度
    pixel_densities = [complexity_summary[c]['pixel_density'] for c in range(10)]
    axes[0, 0].bar(range(10), pixel_densities, color=colors)
    axes[0, 0].set_xticks(range(10))
    axes[0, 0].set_xticklabels(roman_labels)
    axes[0, 0].set_ylabel('Pixel Density')
    axes[0, 0].set_title('Pixel Density by Character')

    # 边缘密度
    edge_densities = [complexity_summary[c]['edge_density'] for c in range(10)]
    axes[0, 1].bar(range(10), edge_densities, color=colors)
    axes[0, 1].set_xticks(range(10))
    axes[0, 1].set_xticklabels(roman_labels)
    axes[0, 1].set_ylabel('Edge Density')
    axes[0, 1].set_title('Edge Density by Character')

    # 综合得分排序
    overall_scores = [complexity_summary[c]['overall'] for c in sorted_classes]
    sorted_labels = [roman_labels[c] for c in sorted_classes]
    axes[1, 0].barh(range(10), overall_scores, color=[colors[c] for c in sorted_classes])
    axes[1, 0].set_yticks(range(10))
    axes[1, 0].set_yticklabels(sorted_labels)
    axes[1, 0].set_xlabel('Overall Complexity Score')
    axes[1, 0].set_title('Character Complexity Ranking')
    axes[1, 0].invert_yaxis()

    # 复杂度雷达图 (简化为柱状对比)
    metrics = ['pixel_density', 'edge_density', 'variance']
    x = np.arange(len(metrics))
    width = 0.08

    for i, cls in enumerate(range(10)):
        values = [complexity_summary[cls][m] for m in metrics]
        # 归一化
        max_vals = [max(complexity_summary[c][m] for c in range(10)) for m in metrics]
        norm_values = [v/m if m > 0 else 0 for v, m in zip(values, max_vals)]
        axes[1, 1].bar(x + i*width, norm_values, width, label=roman_labels[cls], color=colors[i])

    axes[1, 1].set_xticks(x + width * 4.5)
    axes[1, 1].set_xticklabels(['Pixel\nDensity', 'Edge\nDensity', 'Variance'])
    axes[1, 1].set_ylabel('Normalized Value')
    axes[1, 1].set_title('Complexity Metrics Comparison')
    axes[1, 1].legend(loc='upper right', ncol=5, fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/stroke_complexity.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "笔画复杂度分析")


def generate_summary(train_dataset, test_dataset):
    """保存数据统计"""
    summary = f"""
KMNIST 数据统计
---------------
训练: {len(train_dataset)}
测试: {len(test_dataset)}
图像: 28x28 灰度
类别: 10 (平假名)

每类训练样本: 6000
每类测试样本: 1000
"""
    with open(f'{OUTPUT_DIR}/eda_summary.txt', 'w') as f:
        f.write(summary)


def main():
    """主函数"""
    print("KMNIST Dataset - Exploratory Data Analysis")
    print("=" * 60)

    train_dataset, test_dataset = load_data()
    train_labels, test_labels = basic_info(train_dataset, test_dataset)
    class_distribution(train_labels, test_labels)
    sample_visualization(train_dataset)
    pixel_statistics(train_dataset)
    class_similarity_analysis(train_dataset)
    stroke_complexity_analysis(train_dataset)
    generate_summary(train_dataset, test_dataset)

    # 生成HTML报告
    html_content = report.generate_html()
    html_path = f'{OUTPUT_DIR}/eda_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML报告已生成: {html_path}")
    print(f"📊 图片已保存到: {OUTPUT_DIR}/")
    print("\n" + "=" * 60)
    print("EDA 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
