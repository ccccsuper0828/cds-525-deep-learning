"""
CIFAR-100 Dataset - Exploratory Data Analysis
数据集: https://www.cs.toronto.edu/~kriz/cifar.html
PyTorch内置: torchvision.datasets.CIFAR100
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

OUTPUT_DIR = '../reports/figures/cifar100'
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
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #ffffff;
            min-height: 100vh;
            color: #333333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        header {{
            text-align: center;
            padding: 60px 0;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 20px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(240, 147, 251, 0.2);
        }}
        h1 {{
            font-size: 2.5em;
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: rgba(255,255,255,0.9);
            font-size: 1.1em;
        }}
        .section {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid #e9ecef;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }}
        .section h2 {{
            color: #e91e63;
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e91e63;
        }}
        .section pre {{
            background: #f1f3f4;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #333333;
            border: 1px solid #e0e0e0;
        }}
        .section img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            display: block;
            margin: 20px auto;
        }}
        footer {{
            text-align: center;
            padding: 30px;
            color: #666666;
            font-size: 0.9em;
        }}
        .timestamp {{
            color: rgba(255,255,255,0.8);
            font-size: 0.85em;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🖼️ {self.title}</h1>
            <p class="subtitle">Exploratory Data Analysis Report</p>
            <p class="timestamp">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>
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
            <p>CIFAR-100 Dataset - EDA Report</p>
        </footer>
    </div>
</body>
</html>'''
        return html


# 全局报告对象
report = HTMLReport("CIFAR-100 Dataset EDA")


# CIFAR-100 超类和细分类
SUPERCLASSES = {
    0: 'aquatic mammals',
    1: 'fish',
    2: 'flowers',
    3: 'food containers',
    4: 'fruit and vegetables',
    5: 'household electrical devices',
    6: 'household furniture',
    7: 'insects',
    8: 'large carnivores',
    9: 'large man-made outdoor things',
    10: 'large natural outdoor scenes',
    11: 'large omnivores and herbivores',
    12: 'medium-sized mammals',
    13: 'non-insect invertebrates',
    14: 'people',
    15: 'reptiles',
    16: 'small mammals',
    17: 'trees',
    18: 'vehicles 1',
    19: 'vehicles 2'
}

# 部分细分类示例
FINE_CLASSES_SAMPLE = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur'
]


def load_data(data_dir='../data'):
    """加载CIFAR-100数据集"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("1. 数据加载\n")
    output.write("=" * 60 + "\n")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    try:
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform
        )
        output.write("数据集下载/加载成功!\n")
    except Exception as e:
        output.write(f"下载出现问题: {e}\n")
        output.write("尝试继续...\n")
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=False, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=False, transform=transform
        )

    output.write(f"\n训练集大小: {len(train_dataset)}\n")
    output.write(f"测试集大小: {len(test_dataset)}\n")
    output.write(f"图像尺寸: {train_dataset[0][0].shape}\n")
    output.write(f"细分类别数: 100\n")
    output.write(f"超类数: 20\n")

    # 获取类别名称
    if hasattr(train_dataset, 'classes'):
        output.write(f"\n部分类别名称: {train_dataset.classes[:10]}...\n")

    report.add_section("1. 数据加载", output.getvalue())
    return train_dataset, test_dataset


def basic_info(train_dataset, test_dataset):
    """基本信息统计"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("2. 基本信息统计\n")
    output.write("=" * 60 + "\n")

    # 获取标签
    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)

    output.write(f"\n训练集:\n")
    output.write(f"  样本数: {len(train_dataset)}\n")
    output.write(f"  每类样本数: {len(train_dataset) // 100}\n")

    output.write(f"\n测试集:\n")
    output.write(f"  样本数: {len(test_dataset)}\n")
    output.write(f"  每类样本数: {len(test_dataset) // 100}\n")

    output.write(f"\n图像信息:\n")
    sample_image, _ = train_dataset[0]
    output.write(f"  形状: {sample_image.shape} (C, H, W)\n")
    output.write(f"  通道: RGB (3通道)\n")
    output.write(f"  数据类型: {sample_image.dtype}\n")
    output.write(f"  像素值范围: [{sample_image.min():.3f}, {sample_image.max():.3f}]\n")

    report.add_section("2. 基本信息统计", output.getvalue())
    return train_labels, test_labels


def class_distribution(train_labels, test_labels, train_dataset):
    """类别分布分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("3. 类别分布分析\n")
    output.write("=" * 60 + "\n")

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    output.write(f"\n训练集每类样本数: {train_counts[0]} (所有类别相同)\n")
    output.write(f"测试集每类样本数: {test_counts[0]} (所有类别相同)\n")

    # 获取类别名称
    if hasattr(train_dataset, 'classes'):
        class_names = train_dataset.classes
    else:
        class_names = [f'class_{i}' for i in range(100)]

    # 验证平衡性
    balance_ratio = min(train_counts.values()) / max(train_counts.values())
    output.write(f"\n类别平衡比: {balance_ratio:.3f} (1.0 = 完美平衡)\n")

    report.add_section("3. 类别分布分析", output.getvalue())

    # 可视化 - 按超类分组
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 超类分布
    superclass_counts = {i: 0 for i in range(20)}
    # 假设每5个细分类属于一个超类
    for fine_class, count in train_counts.items():
        superclass = fine_class // 5
        superclass_counts[superclass] += count

    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # 超类柱状图
    superclass_names = [SUPERCLASSES[i][:15] for i in range(20)]
    superclass_values = list(superclass_counts.values())
    axes[0, 0].barh(range(20), superclass_values, color=colors)
    axes[0, 0].set_yticks(range(20))
    axes[0, 0].set_yticklabels(superclass_names, fontsize=8)
    axes[0, 0].set_xlabel('Sample Count')
    axes[0, 0].set_title('Training Samples by Superclass')
    axes[0, 0].invert_yaxis()

    # 细分类分布 (前30个)
    fine_counts = [train_counts[i] for i in range(30)]
    fine_names = class_names[:30] if len(class_names) >= 30 else [f'class_{i}' for i in range(30)]
    axes[0, 1].bar(range(30), fine_counts, color='steelblue', alpha=0.7)
    axes[0, 1].set_xticks(range(0, 30, 5))
    axes[0, 1].set_xlabel('Fine Class Index')
    axes[0, 1].set_ylabel('Sample Count')
    axes[0, 1].set_title('Training Samples by Fine Class (First 30)')

    # 训练/测试比例
    train_test_ratio = len(train_labels) / len(test_labels)
    axes[1, 0].bar(['Training', 'Test'], [len(train_labels), len(test_labels)],
                  color=['steelblue', 'coral'])
    axes[1, 0].set_ylabel('Sample Count')
    axes[1, 0].set_title(f'Train/Test Split (Ratio: {train_test_ratio:.1f}:1)')

    # 每类样本数验证
    unique_counts = list(set(train_counts.values()))
    axes[1, 1].bar(['Samples per Class'], unique_counts, color='green')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Samples per Class (All Classes Equal)')
    axes[1, 1].text(0, unique_counts[0] + 10, f'{unique_counts[0]}', ha='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/class_distribution.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "类别分布可视化")


def sample_visualization(train_dataset):
    """样本可视化"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("4. 样本可视化\n")
    output.write("=" * 60 + "\n")
    output.write("展示每个超类的代表性样本图像\n")

    report.add_section("4. 样本可视化", output.getvalue())

    # 获取类别名称
    if hasattr(train_dataset, 'classes'):
        class_names = train_dataset.classes
    else:
        class_names = [f'class_{i}' for i in range(100)]

    targets = np.array(train_dataset.targets)

    # 显示每个超类的一个样本
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for superclass_idx in range(20):
        # 该超类对应的细分类 (假设每5个细分类属于一个超类)
        fine_classes = range(superclass_idx * 5, (superclass_idx + 1) * 5)

        # 找一个样本
        for fine_class in fine_classes:
            indices = np.where(targets == fine_class)[0]
            if len(indices) > 0:
                idx = indices[0]
                break

        image, label = train_dataset[idx]
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)

        ax = axes[superclass_idx]
        ax.imshow(image)
        ax.set_title(f'{SUPERCLASSES[superclass_idx][:20]}', fontsize=9)
        ax.axis('off')

    fig.suptitle('Sample Images from Each Superclass', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/superclass_samples.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "超类样本展示")

    # 显示更多样本 (10x10网格)
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))

    for i in range(100):
        idx = np.where(targets == i)[0][0]
        image, _ = train_dataset[idx]
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)

        ax = axes[i // 10, i % 10]
        ax.imshow(image)
        ax.axis('off')

    fig.suptitle('One Sample from Each of 100 Fine Classes', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/all_classes_samples.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "100个细分类样本展示")


def pixel_and_color_analysis(train_dataset):
    """像素和颜色分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("5. 像素和颜色分析\n")
    output.write("=" * 60 + "\n")

    # 采样分析
    n_samples = 5000

    r_values, g_values, b_values = [], [], []

    for i in range(n_samples):
        image, _ = train_dataset[i]
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        r_values.extend(image[0].flatten())
        g_values.extend(image[1].flatten())
        b_values.extend(image[2].flatten())

    r_values = np.array(r_values)
    g_values = np.array(g_values)
    b_values = np.array(b_values)

    output.write(f"\nRGB通道统计 (基于{n_samples}个样本):\n")
    output.write(f"  R通道 - 均值: {r_values.mean():.4f}, 标准差: {r_values.std():.4f}\n")
    output.write(f"  G通道 - 均值: {g_values.mean():.4f}, 标准差: {g_values.std():.4f}\n")
    output.write(f"  B通道 - 均值: {b_values.mean():.4f}, 标准差: {b_values.std():.4f}\n")

    report.add_section("5. 像素和颜色分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RGB直方图
    axes[0, 0].hist(r_values, bins=50, alpha=0.5, color='red', label='R')
    axes[0, 0].hist(g_values, bins=50, alpha=0.5, color='green', label='G')
    axes[0, 0].hist(b_values, bins=50, alpha=0.5, color='blue', label='B')
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('RGB Channel Distribution')
    axes[0, 0].legend()

    # 各通道箱线图
    bp = axes[0, 1].boxplot([r_values[::100], g_values[::100], b_values[::100]],
                            labels=['R', 'G', 'B'], patch_artist=True)
    colors = ['red', 'green', 'blue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[0, 1].set_ylabel('Pixel Value')
    axes[0, 1].set_title('RGB Channel Box Plot')

    # 平均图像
    avg_image = np.zeros((32, 32, 3))
    for i in range(min(1000, n_samples)):
        image, _ = train_dataset[i]
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)
        avg_image += image / min(1000, n_samples)

    axes[1, 0].imshow(avg_image)
    axes[1, 0].set_title('Average Image (1000 samples)')
    axes[1, 0].axis('off')

    # 像素强度热力图
    intensity = avg_image.mean(axis=2)
    im = axes[1, 1].imshow(intensity, cmap='hot')
    axes[1, 1].set_title('Average Pixel Intensity')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pixel_analysis.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "像素和颜色分析")

    # 返回统计值供归一化使用
    return {
        'mean': [r_values.mean(), g_values.mean(), b_values.mean()],
        'std': [r_values.std(), g_values.std(), b_values.std()]
    }


def class_similarity_analysis(train_dataset):
    """类别相似度分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("6. 类别相似度分析\n")
    output.write("=" * 60 + "\n")

    targets = np.array(train_dataset.targets)
    n_samples_per_class = 100  # 每个类别采样数

    # 计算每个超类的平均图像向量
    superclass_vectors = {}
    for superclass_idx in range(20):
        fine_classes = range(superclass_idx * 5, (superclass_idx + 1) * 5)
        class_images = []

        for fine_class in fine_classes:
            indices = np.where(targets == fine_class)[0][:n_samples_per_class // 5]
            for idx in indices:
                image, _ = train_dataset[idx]
                if isinstance(image, torch.Tensor):
                    image = image.numpy()
                class_images.append(image.flatten())

        if class_images:
            superclass_vectors[superclass_idx] = np.mean(class_images, axis=0)

    # 计算超类间余弦相似度
    similarity_matrix = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            if i in superclass_vectors and j in superclass_vectors:
                dot = np.dot(superclass_vectors[i], superclass_vectors[j])
                norm_i = np.linalg.norm(superclass_vectors[i])
                norm_j = np.linalg.norm(superclass_vectors[j])
                similarity_matrix[i, j] = dot / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0

    # 找出最相似的超类对
    similar_pairs = []
    for i in range(20):
        for j in range(i+1, 20):
            similar_pairs.append((i, j, similarity_matrix[i, j]))
    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    output.write("\n最相似的超类对 (余弦相似度):\n")
    for i, j, sim in similar_pairs[:5]:
        output.write(f"  {SUPERCLASSES[i]} vs {SUPERCLASSES[j]}: {sim:.4f}\n")

    output.write("\n最不相似的超类对:\n")
    for i, j, sim in similar_pairs[-5:]:
        output.write(f"  {SUPERCLASSES[i]} vs {SUPERCLASSES[j]}: {sim:.4f}\n")

    report.add_section("6. 类别相似度分析", output.getvalue())

    # 可视化相似度矩阵
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(similarity_matrix, cmap='YlOrRd', vmin=0.5, vmax=1)
    ax.set_xticks(range(20))
    ax.set_yticks(range(20))
    ax.set_xticklabels([SUPERCLASSES[i][:10] for i in range(20)], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([SUPERCLASSES[i][:10] for i in range(20)], fontsize=8)
    ax.set_title('Superclass Similarity Matrix (Cosine Similarity)')
    plt.colorbar(im)

    # 添加数值标签
    for i in range(20):
        for j in range(20):
            ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=6)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/class_similarity.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "类别相似度矩阵")


def brightness_contrast_analysis(train_dataset):
    """亮度和对比度分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("7. 亮度和对比度分析\n")
    output.write("=" * 60 + "\n")

    n_samples = 5000
    brightness_values = []
    contrast_values = []

    for i in range(n_samples):
        image, _ = train_dataset[i]
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        # 转换为灰度计算亮度 (使用标准权重)
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]

        # 亮度 = 平均像素值
        brightness = gray.mean()
        brightness_values.append(brightness)

        # 对比度 = 像素值标准差
        contrast = gray.std()
        contrast_values.append(contrast)

    brightness_values = np.array(brightness_values)
    contrast_values = np.array(contrast_values)

    output.write(f"\n亮度统计 (基于{n_samples}个样本):\n")
    output.write(f"  平均亮度: {brightness_values.mean():.4f}\n")
    output.write(f"  亮度标准差: {brightness_values.std():.4f}\n")
    output.write(f"  最低亮度: {brightness_values.min():.4f}\n")
    output.write(f"  最高亮度: {brightness_values.max():.4f}\n")

    output.write(f"\n对比度统计:\n")
    output.write(f"  平均对比度: {contrast_values.mean():.4f}\n")
    output.write(f"  对比度标准差: {contrast_values.std():.4f}\n")
    output.write(f"  最低对比度: {contrast_values.min():.4f}\n")
    output.write(f"  最高对比度: {contrast_values.max():.4f}\n")

    report.add_section("7. 亮度和对比度分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 亮度分布
    axes[0, 0].hist(brightness_values, bins=50, color='gold', edgecolor='white', alpha=0.7)
    axes[0, 0].axvline(brightness_values.mean(), color='red', linestyle='--',
                       label=f'Mean: {brightness_values.mean():.3f}')
    axes[0, 0].set_xlabel('Brightness')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Brightness Distribution')
    axes[0, 0].legend()

    # 对比度分布
    axes[0, 1].hist(contrast_values, bins=50, color='purple', edgecolor='white', alpha=0.7)
    axes[0, 1].axvline(contrast_values.mean(), color='red', linestyle='--',
                       label=f'Mean: {contrast_values.mean():.3f}')
    axes[0, 1].set_xlabel('Contrast')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Contrast Distribution')
    axes[0, 1].legend()

    # 亮度 vs 对比度散点图
    axes[1, 0].scatter(brightness_values, contrast_values, alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Brightness')
    axes[1, 0].set_ylabel('Contrast')
    axes[1, 0].set_title('Brightness vs Contrast')

    # 按超类分组的亮度/对比度
    targets = np.array(train_dataset.targets)
    superclass_brightness = {i: [] for i in range(20)}

    for i in range(n_samples):
        superclass = targets[i] // 5
        superclass_brightness[superclass].append(brightness_values[i])

    superclass_means = [np.mean(superclass_brightness[i]) for i in range(20)]
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    axes[1, 1].barh(range(20), superclass_means, color=colors)
    axes[1, 1].set_yticks(range(20))
    axes[1, 1].set_yticklabels([SUPERCLASSES[i][:12] for i in range(20)], fontsize=8)
    axes[1, 1].set_xlabel('Average Brightness')
    axes[1, 1].set_title('Average Brightness by Superclass')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/brightness_contrast.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "亮度和对比度分析")


def superclass_analysis(train_dataset):
    """超类分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("8. 超类详细分析\n")
    output.write("=" * 60 + "\n")

    output.write("\n超类及其包含的细分类:\n")
    for superclass_idx, superclass_name in SUPERCLASSES.items():
        fine_start = superclass_idx * 5
        fine_end = fine_start + 5

        if hasattr(train_dataset, 'classes'):
            fine_names = train_dataset.classes[fine_start:fine_end]
        else:
            fine_names = [f'class_{i}' for i in range(fine_start, fine_end)]

        output.write(f"\n  {superclass_name}:\n")
        output.write(f"    细分类: {fine_names}\n")

    report.add_section("6. 超类详细分析", output.getvalue())

    # 可视化超类关系
    fig, ax = plt.subplots(figsize=(12, 10))

    # 创建超类-细分类关系矩阵
    matrix = np.zeros((20, 100))
    for superclass_idx in range(20):
        for fine_idx in range(5):
            matrix[superclass_idx, superclass_idx * 5 + fine_idx] = 1

    ax.imshow(matrix, cmap='Blues', aspect='auto')
    ax.set_xlabel('Fine Class Index')
    ax.set_ylabel('Superclass')
    ax.set_yticks(range(20))
    ax.set_yticklabels([SUPERCLASSES[i][:15] for i in range(20)], fontsize=8)
    ax.set_title('Superclass to Fine Class Mapping')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/superclass_mapping.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "超类映射关系")


def generate_summary(train_dataset, test_dataset, pixel_stats):
    """生成EDA总结"""
    summary = f"""
CIFAR-100 Dataset EDA Summary
=============================

1. Dataset Overview:
   - Training samples: {len(train_dataset)}
   - Test samples: {len(test_dataset)}
   - Image size: 32 x 32 RGB
   - Fine classes: 100
   - Superclasses: 20

2. Class Structure:
   - Each superclass contains 5 fine classes
   - Training: 500 samples per fine class
   - Test: 100 samples per fine class
   - Perfectly balanced dataset

3. Image Statistics:
   - Pixel value range: [0, 1]
   - R channel - Mean: {pixel_stats['mean'][0]:.4f}, Std: {pixel_stats['std'][0]:.4f}
   - G channel - Mean: {pixel_stats['mean'][1]:.4f}, Std: {pixel_stats['std'][1]:.4f}
   - B channel - Mean: {pixel_stats['mean'][2]:.4f}, Std: {pixel_stats['std'][2]:.4f}

4. Key Challenges:
   - Fine-grained classification (100 classes)
   - Low resolution (32x32)
   - High intra-class variance
   - Some visually similar classes

5. Recommended Models:
   - ResNet-56/110: Good baseline (~72-75%)
   - WideResNet: Strong performance (~80-82%)
   - DenseNet: Competitive results (~78-82%)
   - Vision Transformer: State-of-the-art (~85-90%)

6. Preprocessing Suggestions:
   - Normalize with dataset mean/std
   - Data augmentation: random crop, horizontal flip
   - Cutout/MixUp for regularization

7. Recommended Normalization:
   - Mean: [{pixel_stats['mean'][0]:.4f}, {pixel_stats['mean'][1]:.4f}, {pixel_stats['mean'][2]:.4f}]
   - Std: [{pixel_stats['std'][0]:.4f}, {pixel_stats['std'][1]:.4f}, {pixel_stats['std'][2]:.4f}]

8. Evaluation:
   - Primary metric: Top-1 accuracy on fine classes
   - Secondary: Top-5 accuracy, superclass accuracy
"""
    report.add_section("9. EDA 总结报告", summary)

    with open(f'{OUTPUT_DIR}/eda_summary.txt', 'w') as f:
        f.write(summary)


def main():
    """主函数"""
    print("CIFAR-100 Dataset - Exploratory Data Analysis")
    print("=" * 60)

    train_dataset, test_dataset = load_data()
    train_labels, test_labels = basic_info(train_dataset, test_dataset)
    class_distribution(train_labels, test_labels, train_dataset)
    sample_visualization(train_dataset)
    pixel_stats = pixel_and_color_analysis(train_dataset)
    class_similarity_analysis(train_dataset)
    brightness_contrast_analysis(train_dataset)
    superclass_analysis(train_dataset)
    generate_summary(train_dataset, test_dataset, pixel_stats)

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
