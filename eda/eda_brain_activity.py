"""
Harmful Brain Activity Classification Dataset - Exploratory Data Analysis
数据集: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification
输出: HTML报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import base64
from io import BytesIO, StringIO
import sys
from datetime import datetime

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = '../reports/figures/brain_activity'
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
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
            color: #667eea;
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
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
            <h1>🧠 {self.title}</h1>
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
            <p>Harmful Brain Activity Classification - EDA Report</p>
        </footer>
    </div>
</body>
</html>'''
        return html


# 全局报告对象
report = HTMLReport("Harmful Brain Activity Classification Dataset EDA")


# 脑活动类别定义
CLASSES = {
    'seizure': 'Seizure (癫痫发作)',
    'lpd': 'LPD - Lateralized Periodic Discharges',
    'gpd': 'GPD - Generalized Periodic Discharges',
    'lrda': 'LRDA - Lateralized Rhythmic Delta Activity',
    'grda': 'GRDA - Generalized Rhythmic Delta Activity',
    'other': 'Other (其他)'
}

# EEG通道名称 (标准10-20系统)
EEG_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
    'EKG'
]


def load_metadata(data_dir):
    """加载元数据"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("1. 数据加载与概览\n")
    output.write("=" * 60 + "\n")

    train_csv = Path(data_dir) / 'train.csv'

    if not train_csv.exists():
        output.write(f"[WARNING] 找不到 {train_csv}\n")
        output.write("请从Kaggle下载数据集:\n")
        output.write("https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data\n")
        output.write("\n使用演示数据进行分析...\n")
        df = create_demo_data()
    else:
        df = pd.read_csv(train_csv)
    
    output.write(f"训练集形状: {df.shape}\n")
    output.write(f"列名: {df.columns.tolist()}\n")
    
    report.add_section("1. 数据加载与概览", output.getvalue())
    return df


def create_demo_data():
    """创建演示数据"""
    np.random.seed(42)
    n_samples = 1000

    demo_data = {
        'eeg_id': range(n_samples),
        'patient_id': np.random.randint(1, 100, n_samples),
        'seizure_vote': np.random.randint(0, 10, n_samples),
        'lpd_vote': np.random.randint(0, 10, n_samples),
        'gpd_vote': np.random.randint(0, 10, n_samples),
        'lrda_vote': np.random.randint(0, 10, n_samples),
        'grda_vote': np.random.randint(0, 10, n_samples),
        'other_vote': np.random.randint(0, 10, n_samples),
    }

    df = pd.DataFrame(demo_data)

    # 归一化投票为概率
    vote_cols = [c for c in df.columns if '_vote' in c]
    vote_sum = df[vote_cols].sum(axis=1)
    for col in vote_cols:
        df[col] = df[col] / vote_sum

    return df


def basic_statistics(df):
    """基本统计信息"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("2. 基本统计信息\n")
    output.write("=" * 60 + "\n")

    output.write(f"\n总样本数: {len(df)}\n")
    output.write(f"特征数量: {df.shape[1]}\n")

    if 'patient_id' in df.columns:
        output.write(f"患者数量: {df['patient_id'].nunique()}\n")

    output.write(f"\n缺失值统计:\n{df.isnull().sum()}\n")
    output.write(f"\n数值列统计:\n{df.describe()}\n")
    
    report.add_section("2. 基本统计信息", output.getvalue())
    return df


def label_distribution(df):
    """标签分布分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("3. 标签分布分析\n")
    output.write("=" * 60 + "\n")

    # 找到投票列
    vote_cols = [c for c in df.columns if '_vote' in c or c in CLASSES.keys()]

    if not vote_cols:
        vote_cols = [c for c in df.columns if any(cls in c.lower() for cls in CLASSES.keys())]

    if not vote_cols:
        output.write("未找到投票/标签列，跳过分布分析\n")
        report.add_section("3. 标签分布分析", output.getvalue())
        return

    output.write(f"投票列: {vote_cols}\n")

    # 计算每个类别的平均投票
    avg_votes = df[vote_cols].mean()
    output.write(f"\n各类别平均投票/概率:\n{avg_votes}\n")

    # 基于最高投票确定主要类别
    if len(vote_cols) > 1:
        df['primary_class'] = df[vote_cols].idxmax(axis=1)
        class_counts = df['primary_class'].value_counts()
        output.write(f"\n主要类别分布:\n{class_counts}\n")
    
    report.add_section("3. 标签分布分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 平均投票柱状图
    colors = plt.cm.Set3(np.linspace(0, 1, len(vote_cols)))
    axes[0, 0].bar(range(len(avg_votes)), avg_votes.values, color=colors)
    axes[0, 0].set_xticks(range(len(avg_votes)))
    axes[0, 0].set_xticklabels([c.replace('_vote', '') for c in vote_cols], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Average Vote / Probability')
    axes[0, 0].set_title('Average Vote Distribution by Class')

    # 主要类别饼图
    if 'primary_class' in df.columns:
        class_counts = df['primary_class'].value_counts()
        axes[0, 1].pie(class_counts.values,
                      labels=[c.replace('_vote', '') for c in class_counts.index],
                      autopct='%1.1f%%', colors=colors[:len(class_counts)])
        axes[0, 1].set_title('Primary Class Distribution')

    # 投票分布箱线图
    vote_data = [df[col].dropna() for col in vote_cols]
    bp = axes[1, 0].boxplot(vote_data, labels=[c.replace('_vote', '') for c in vote_cols],
                            patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel('Vote / Probability')
    axes[1, 0].set_title('Vote Distribution by Class')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 投票相关性热力图
    corr = df[vote_cols].corr()
    im = axes[1, 1].imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(vote_cols)))
    axes[1, 1].set_yticks(range(len(vote_cols)))
    axes[1, 1].set_xticklabels([c.replace('_vote', '') for c in vote_cols], rotation=45, ha='right')
    axes[1, 1].set_yticklabels([c.replace('_vote', '') for c in vote_cols])
    axes[1, 1].set_title('Vote Correlation Matrix')
    plt.colorbar(im, ax=axes[1, 1])

    # 添加相关系数数值
    for i in range(len(vote_cols)):
        for j in range(len(vote_cols)):
            axes[1, 1].text(j, i, f'{corr.iloc[i, j]:.2f}',
                           ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/label_distribution.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "标签分布可视化")


def patient_analysis(df):
    """患者级别分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("4. 患者级别分析\n")
    output.write("=" * 60 + "\n")

    if 'patient_id' not in df.columns:
        output.write("未找到patient_id列，跳过患者分析\n")
        report.add_section("4. 患者级别分析", output.getvalue())
        return

    # 每个患者的样本数
    samples_per_patient = df['patient_id'].value_counts()
    output.write(f"\n每患者样本数统计:\n")
    output.write(f"  平均: {samples_per_patient.mean():.2f}\n")
    output.write(f"  最少: {samples_per_patient.min()}\n")
    output.write(f"  最多: {samples_per_patient.max()}\n")
    output.write(f"  中位数: {samples_per_patient.median():.0f}\n")
    
    report.add_section("4. 患者级别分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 每患者样本数分布
    axes[0].hist(samples_per_patient.values, bins=30, color='steelblue',
                edgecolor='white', alpha=0.7)
    axes[0].axvline(samples_per_patient.mean(), color='red', linestyle='--',
                   label=f'Mean: {samples_per_patient.mean():.1f}')
    axes[0].set_xlabel('Samples per Patient')
    axes[0].set_ylabel('Number of Patients')
    axes[0].set_title('Distribution of Samples per Patient')
    axes[0].legend()

    # Top 20 患者
    top_patients = samples_per_patient.head(20)
    axes[1].barh(range(len(top_patients)), top_patients.values, color='coral')
    axes[1].set_yticks(range(len(top_patients)))
    axes[1].set_yticklabels([f'Patient {p}' for p in top_patients.index])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Number of Samples')
    axes[1].set_title('Top 20 Patients by Sample Count')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/patient_analysis.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "患者级别分析")


def simulate_eeg_visualization():
    """模拟EEG信号可视化 (演示用)"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("5. EEG信号可视化示例\n")
    output.write("=" * 60 + "\n")
    output.write("\n注意: 这是模拟的EEG数据，用于展示可视化方法\n")
    output.write("实际数据需要从Kaggle下载parquet文件\n")
    
    report.add_section("5. EEG信号可视化示例", output.getvalue())

    # 生成模拟EEG数据
    np.random.seed(42)
    duration = 10  # 秒
    fs = 200  # 采样率
    t = np.linspace(0, duration, duration * fs)

    n_channels = 8
    channel_names = EEG_CHANNELS[:n_channels]

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 12), sharex=True)

    for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        # 模拟不同频率成分
        alpha = np.sin(2 * np.pi * 10 * t) * np.random.uniform(0.5, 1.5)  # Alpha波
        beta = np.sin(2 * np.pi * 20 * t) * np.random.uniform(0.2, 0.5)   # Beta波
        noise = np.random.normal(0, 0.3, len(t))

        signal = alpha + beta + noise

        ax.plot(t, signal, 'b-', linewidth=0.5)
        ax.set_ylabel(ch_name)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle('Simulated EEG Signal Visualization (8 Channels)', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/eeg_visualization.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "EEG信号可视化")


def class_description():
    """类别详细说明"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("6. 脑活动类别说明\n")
    output.write("=" * 60 + "\n")

    descriptions = {
        'Seizure': '癫痫发作 - 大脑异常放电导致的发作性症状',
        'LPD': '侧向周期性放电 - 一侧大脑的周期性电活动异常',
        'GPD': '广泛性周期性放电 - 双侧大脑的周期性电活动异常',
        'LRDA': '侧向节律性δ活动 - 一侧大脑的节律性慢波活动',
        'GRDA': '广泛性节律性δ活动 - 双侧大脑的节律性慢波活动',
        'Other': '其他 - 不属于以上类别的异常活动'
    }

    output.write("\n各类别临床意义:\n")
    for cls, desc in descriptions.items():
        output.write(f"\n  {cls}:\n")
        output.write(f"    {desc}\n")
    
    report.add_section("6. 脑活动类别说明", output.getvalue())

    # 可视化类别关系
    fig, ax = plt.subplots(figsize=(10, 8))

    # 创建简单的类别关系图
    categories = list(descriptions.keys())
    y_positions = range(len(categories))

    ax.barh(y_positions, [1]*len(categories), color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel('')
    ax.set_title('Brain Activity Classification Categories')

    # 添加描述文本
    for i, (cat, desc) in enumerate(descriptions.items()):
        ax.text(0.05, i, desc[:50] + '...' if len(desc) > 50 else desc,
               va='center', fontsize=9)

    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/class_description.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "类别说明图")


def generate_summary(df):
    """生成EDA总结"""
    vote_cols = [c for c in df.columns if '_vote' in c]

    summary = f"""
Harmful Brain Activity Classification Dataset EDA Summary
==========================================================

1. Dataset Overview:
   - Total EEG samples: {len(df)}
   - Features: {df.shape[1]}
   - Patients: {df['patient_id'].nunique() if 'patient_id' in df.columns else 'N/A'}

2. Classification Task:
   - Type: Multi-label / Soft-label classification
   - Classes: 6 (Seizure, LPD, GPD, LRDA, GRDA, Other)
   - Labels: Probability distribution from expert votes

3. Data Characteristics:
   - Input: Multi-channel EEG time series
   - Channels: ~20 (standard 10-20 system)
   - Sampling rate: Typically 200 Hz
   - Duration: 50 seconds per sample (typical)

4. Key Challenges:
   - High-dimensional time series data
   - Multi-label soft targets
   - Class imbalance
   - Requires domain expertise for feature engineering

5. Recommended Approaches:
   - 1D CNN for local pattern detection
   - LSTM/GRU for temporal dependencies
   - Transformer for global attention
   - Spectrogram + 2D CNN (convert to image)
   - EfficientNet with spectrogram input

6. Preprocessing Suggestions:
   - Band-pass filtering (0.5-50 Hz)
   - Notch filter for power line noise (50/60 Hz)
   - Z-score normalization per channel
   - Artifact removal (eye blinks, muscle)

7. Evaluation Metric:
   - KL Divergence (competition metric)
   - Also consider: AUC-ROC, Weighted accuracy
"""
    report.add_section("7. EDA 总结报告", summary)

    with open(f'{OUTPUT_DIR}/eda_summary.txt', 'w') as f:
        f.write(summary)


def main():
    """主函数"""
    print("Harmful Brain Activity Classification Dataset - EDA")
    print("=" * 60)

    DATA_DIR = '../data/hms-harmful-brain-activity-classification'

    df = load_metadata(DATA_DIR)
    basic_statistics(df)
    label_distribution(df)
    patient_analysis(df)
    simulate_eeg_visualization()
    class_description()
    generate_summary(df)

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
