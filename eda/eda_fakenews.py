"""
Fake News Detection Dataset - Exploratory Data Analysis
数据集: https://www.kaggle.com/datasets/iamrahulthorat/fakenews-csv
输出: HTML报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
import base64
from io import BytesIO, StringIO
from datetime import datetime

# 设置中文字体和风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 创建输出目录
OUTPUT_DIR = '../reports/figures/fakenews'
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
            <p>Fake News Detection - EDA Report</p>
        </footer>
    </div>
</body>
</html>'''
        return html


# 全局报告对象
report = HTMLReport("Fake News Detection Dataset EDA")


def load_data(filepath):
    """加载数据集"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("1. 数据加载\n")
    output.write("=" * 60 + "\n")

    df = pd.read_csv(filepath)
    output.write(f"数据集形状: {df.shape}\n")
    output.write(f"列名: {df.columns.tolist()}\n")
    output.write(f"\n数据类型:\n{df.dtypes}\n")
    
    report.add_section("1. 数据加载", output.getvalue())
    return df


def basic_info(df):
    """基本信息统计"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("2. 基本信息统计\n")
    output.write("=" * 60 + "\n")

    output.write(f"\n总样本数: {len(df)}\n")
    output.write(f"特征数量: {df.shape[1]}\n")
    output.write(f"\n缺失值统计:\n{df.isnull().sum()}\n")
    output.write(f"\n重复行数: {df.duplicated().sum()}\n")
    output.write(f"\n前5行数据:\n{df.head()}\n")
    
    report.add_section("2. 基本信息统计", output.getvalue())
    return df


def label_distribution(df, label_col='label'):
    """标签分布分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("3. 标签分布分析\n")
    output.write("=" * 60 + "\n")

    # 尝试找到标签列
    possible_label_cols = ['label', 'Label', 'class', 'Class', 'target', 'fake']
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col not in df.columns:
        # 如果找不到，使用最后一列
        label_col = df.columns[-1]

    output.write(f"使用标签列: {label_col}\n")

    label_counts = df[label_col].value_counts()
    output.write(f"\n标签分布:\n{label_counts}\n")
    output.write(f"\n标签比例:\n")
    for label, count in label_counts.items():
        output.write(f"  {label}: {count/len(df)*100:.2f}%\n")

    report.add_section("3. 标签分布分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 柱状图
    colors = ['#2ecc71', '#e74c3c']
    ax1 = axes[0]
    bars = ax1.bar(label_counts.index.astype(str), label_counts.values, color=colors[:len(label_counts)])
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    ax1.set_title('Label Distribution (Bar Chart)')
    for bar, count in zip(bars, label_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha='center', va='bottom', fontsize=12)

    # 饼图
    ax2 = axes[1]
    ax2.pie(label_counts.values, labels=label_counts.index.astype(str),
            autopct='%1.1f%%', colors=colors[:len(label_counts)],
            explode=[0.02] * len(label_counts))
    ax2.set_title('Label Distribution (Pie Chart)')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/label_distribution.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "标签分布可视化")

    return label_col


def text_analysis(df, text_col='text'):
    """文本特征分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("4. 文本特征分析\n")
    output.write("=" * 60 + "\n")

    # 尝试找到文本列
    possible_text_cols = ['text', 'Text', 'content', 'Content', 'news', 'News', 'title', 'Title']
    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col not in df.columns:
        text_col = df.columns[0]

    output.write(f"使用文本列: {text_col}\n")

    # 计算文本长度
    df['text_length'] = df[text_col].astype(str).apply(len)
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))

    output.write(f"\n文本长度统计:\n")
    output.write(f"  平均字符数: {df['text_length'].mean():.2f}\n")
    output.write(f"  最小字符数: {df['text_length'].min()}\n")
    output.write(f"  最大字符数: {df['text_length'].max()}\n")
    output.write(f"  中位数: {df['text_length'].median()}\n")

    output.write(f"\n词数统计:\n")
    output.write(f"  平均词数: {df['word_count'].mean():.2f}\n")
    output.write(f"  最小词数: {df['word_count'].min()}\n")
    output.write(f"  最大词数: {df['word_count'].max()}\n")

    report.add_section("4. 文本特征分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 文本长度分布
    axes[0, 0].hist(df['text_length'], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0, 0].set_xlabel('Character Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Text Length (Characters)')
    axes[0, 0].axvline(df['text_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["text_length"].mean():.0f}')
    axes[0, 0].legend()

    # 词数分布
    axes[0, 1].hist(df['word_count'], bins=50, color='coral', edgecolor='white', alpha=0.7)
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Word Count')
    axes[0, 1].axvline(df['word_count'].mean(), color='red', linestyle='--', label=f'Mean: {df["word_count"].mean():.0f}')
    axes[0, 1].legend()

    # 箱线图
    axes[1, 0].boxplot([df['text_length']], vert=True)
    axes[1, 0].set_ylabel('Character Count')
    axes[1, 0].set_title('Text Length Box Plot')

    axes[1, 1].boxplot([df['word_count']], vert=True)
    axes[1, 1].set_ylabel('Word Count')
    axes[1, 1].set_title('Word Count Box Plot')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/text_length_analysis.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "文本长度分析")

    return text_col


def text_by_label(df, text_col, label_col):
    """按标签分组的文本分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("5. 按标签分组的文本分析\n")
    output.write("=" * 60 + "\n")

    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        output.write(f"\n标签 '{label}':\n")
        output.write(f"  样本数: {len(subset)}\n")
        output.write(f"  平均文本长度: {subset['text_length'].mean():.2f}\n")
        output.write(f"  平均词数: {subset['word_count'].mean():.2f}\n")

    report.add_section("5. 按标签分组的文本分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 按标签分组的文本长度
    labels = df[label_col].unique()
    data_length = [df[df[label_col] == l]['text_length'] for l in labels]
    data_words = [df[df[label_col] == l]['word_count'] for l in labels]

    bp1 = axes[0].boxplot(data_length, labels=[str(l) for l in labels], patch_artist=True)
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp1['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Character Count')
    axes[0].set_title('Text Length by Label')

    bp2 = axes[1].boxplot(data_words, labels=[str(l) for l in labels], patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Word Count')
    axes[1].set_title('Word Count by Label')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/text_by_label.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "按标签分组的文本分析")


def word_frequency_analysis(df, text_col, label_col, top_n=20):
    """词频分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("6. 词频分析\n")
    output.write("=" * 60 + "\n")

    # 简单的分词和清洗
    def tokenize(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        # 过滤停用词
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                    'we', 'they', 'what', 'which', 'who', 'whom', 'how', 'when', 'where',
                    'why', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
                    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                    'too', 'very', 's', 't', 'just', 'don', 'now', 'as', 'said', 'also'}
        return [w for w in words if w not in stopwords and len(w) > 2]

    # 整体词频
    all_words = []
    for text in df[text_col]:
        all_words.extend(tokenize(text))

    word_freq = Counter(all_words)
    top_words = word_freq.most_common(top_n)

    output.write(f"\n整体Top {top_n}高频词:\n")
    for word, count in top_words:
        output.write(f"  {word}: {count}\n")

    report.add_section("6. 词频分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 整体词频
    words, counts = zip(*top_words)
    ax1 = axes[0]
    bars = ax1.barh(range(len(words)), counts, color='steelblue')
    ax1.set_yticks(range(len(words)))
    ax1.set_yticklabels(words)
    ax1.invert_yaxis()
    ax1.set_xlabel('Frequency')
    ax1.set_title(f'Top {top_n} Most Common Words (Overall)')

    # 按标签分组的词频对比
    ax2 = axes[1]
    labels = df[label_col].unique()
    x = np.arange(min(10, top_n))
    width = 0.35

    for i, label in enumerate(labels[:2]):
        subset_words = []
        for text in df[df[label_col] == label][text_col]:
            subset_words.extend(tokenize(text))
        subset_freq = Counter(subset_words).most_common(10)
        if subset_freq:
            _, subset_counts = zip(*subset_freq)
            offset = width * (i - 0.5)
            ax2.bar(x + offset, subset_counts[:10], width, label=str(label))

    ax2.set_xticks(x)
    ax2.set_xticklabels([w for w, _ in top_words[:10]], rotation=45, ha='right')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Word Frequency Comparison by Label')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/word_frequency.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "词频分析")


def ngram_analysis(df, text_col, label_col, top_n=15):
    """N-gram分析 (Bigram和Trigram)"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("7. N-gram分析\n")
    output.write("=" * 60 + "\n")

    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                'we', 'they', 'what', 'which', 'who', 'whom', 'how', 'when', 'where',
                'why', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                'too', 'very', 's', 't', 'just', 'don', 'now', 'as', 'said', 'also'}

    def get_ngrams(text, n):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = [w for w in text.split() if w not in stopwords and len(w) > 2]
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    # 整体Bigram分析
    all_bigrams = []
    all_trigrams = []
    for text in df[text_col]:
        all_bigrams.extend(get_ngrams(text, 2))
        all_trigrams.extend(get_ngrams(text, 3))

    bigram_freq = Counter(all_bigrams).most_common(top_n)
    trigram_freq = Counter(all_trigrams).most_common(top_n)

    output.write(f"\n整体Top {top_n} Bigrams:\n")
    for gram, count in bigram_freq[:10]:
        output.write(f"  '{gram}': {count}\n")

    output.write(f"\n整体Top {top_n} Trigrams:\n")
    for gram, count in trigram_freq[:10]:
        output.write(f"  '{gram}': {count}\n")

    report.add_section("7. N-gram分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 整体Bigram
    if bigram_freq:
        grams, counts = zip(*bigram_freq)
        axes[0, 0].barh(range(len(grams)), counts, color='steelblue')
        axes[0, 0].set_yticks(range(len(grams)))
        axes[0, 0].set_yticklabels(grams, fontsize=9)
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].set_title(f'Top {top_n} Bigrams (Overall)')

    # 整体Trigram
    if trigram_freq:
        grams, counts = zip(*trigram_freq)
        axes[0, 1].barh(range(len(grams)), counts, color='coral')
        axes[0, 1].set_yticks(range(len(grams)))
        axes[0, 1].set_yticklabels(grams, fontsize=9)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_title(f'Top {top_n} Trigrams (Overall)')

    # 按标签分组的Bigram对比
    labels = df[label_col].unique()[:2]
    colors = ['#2ecc71', '#e74c3c']

    for idx, label in enumerate(labels):
        subset_bigrams = []
        for text in df[df[label_col] == label][text_col]:
            subset_bigrams.extend(get_ngrams(text, 2))
        subset_freq = Counter(subset_bigrams).most_common(10)

        if subset_freq:
            grams, counts = zip(*subset_freq)
            y_pos = np.arange(len(grams))
            axes[1, idx].barh(y_pos, counts, color=colors[idx])
            axes[1, idx].set_yticks(y_pos)
            axes[1, idx].set_yticklabels(grams, fontsize=9)
            axes[1, idx].invert_yaxis()
            axes[1, idx].set_xlabel('Frequency')
            axes[1, idx].set_title(f'Top Bigrams - {label}')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ngram_analysis.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "N-gram分析")


def sample_texts_display(df, text_col, label_col, n=5):
    """样本文本展示"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("8. 样本文本展示\n")
    output.write("=" * 60 + "\n")

    for label in df[label_col].unique():
        output.write(f"\n--- {label} 样本 ---\n")
        samples = df[df[label_col] == label][text_col].head(n)
        for i, text in enumerate(samples, 1):
            text_preview = str(text)[:200].replace('\n', ' ')
            output.write(f"[{i}] {text_preview}...\n\n")

    report.add_section("8. 样本文本展示", output.getvalue())


def wordcloud_analysis(df, text_col, label_col):
    """词云可视化"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("9. 词云分析\n")
    output.write("=" * 60 + "\n")

    try:
        from wordcloud import WordCloud
    except ImportError:
        output.write("WordCloud未安装，跳过词云分析\n")
        output.write("安装: pip install wordcloud\n")
        report.add_section("9. 词云分析", output.getvalue())
        return

    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                'have', 'has', 'had', 'it', 'its', 'this', 'that', 'said', 'would',
                'could', 'will', 'can', 'may', 'also', 'as', 'just', 'not', 'more'}

    labels = df[label_col].unique()
    fig, axes = plt.subplots(1, len(labels), figsize=(7*len(labels), 6))
    if len(labels) == 1:
        axes = [axes]

    colors = {'0': 'Reds', '1': 'Greens', 'Fake': 'Reds', 'Real': 'Greens',
              'fake': 'Reds', 'real': 'Greens'}

    for idx, label in enumerate(labels):
        text = ' '.join(df[df[label_col] == label][text_col].astype(str).tolist())
        wc = WordCloud(width=800, height=400, background_color='white',
                      stopwords=stopwords, max_words=100,
                      colormap=colors.get(str(label), 'viridis'))
        wc.generate(text)

        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'Word Cloud - {label}', fontsize=14)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/wordcloud.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "词云分析")

    output.write("词云已生成\n")
    report.add_section("9. 词云分析", output.getvalue())


def text_quality_analysis(df, text_col, label_col):
    """文本质量分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("10. 文本质量分析\n")
    output.write("=" * 60 + "\n")

    # 计算各种质量指标
    df['has_special_chars'] = df[text_col].astype(str).apply(
        lambda x: bool(re.search(r'[^\w\s.,!?\'\"-]', x))
    )
    df['is_empty'] = df[text_col].astype(str).apply(lambda x: len(x.strip()) == 0)
    df['is_short'] = df['word_count'] < 5
    df['has_url'] = df[text_col].astype(str).apply(lambda x: bool(re.search(r'http\S+|www\.\S+', x)))
    df['has_numbers'] = df[text_col].astype(str).apply(lambda x: bool(re.search(r'\d+', x)))
    df['uppercase_ratio'] = df[text_col].astype(str).apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    df['avg_word_length'] = df[text_col].astype(str).apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )

    output.write("\n文本质量统计:\n")
    output.write(f"  空文本数量: {df['is_empty'].sum()} ({df['is_empty'].mean()*100:.2f}%)\n")
    output.write(f"  短文本数量 (<5词): {df['is_short'].sum()} ({df['is_short'].mean()*100:.2f}%)\n")
    output.write(f"  含特殊字符: {df['has_special_chars'].sum()} ({df['has_special_chars'].mean()*100:.2f}%)\n")
    output.write(f"  含URL链接: {df['has_url'].sum()} ({df['has_url'].mean()*100:.2f}%)\n")
    output.write(f"  含数字: {df['has_numbers'].sum()} ({df['has_numbers'].mean()*100:.2f}%)\n")
    output.write(f"  平均大写字母比例: {df['uppercase_ratio'].mean()*100:.2f}%\n")
    output.write(f"  平均词长: {df['avg_word_length'].mean():.2f} 字符\n")

    # 按标签分组统计
    output.write("\n按标签分组的质量统计:\n")
    for label in df[label_col].unique():
        subset = df[df[label_col] == label]
        output.write(f"\n  {label}:\n")
        output.write(f"    空文本: {subset['is_empty'].sum()}\n")
        output.write(f"    短文本: {subset['is_short'].sum()}\n")
        output.write(f"    含URL: {subset['has_url'].sum()}\n")
        output.write(f"    大写比例: {subset['uppercase_ratio'].mean()*100:.2f}%\n")

    report.add_section("8. 文本质量分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 质量指标统计
    quality_metrics = ['is_empty', 'is_short', 'has_special_chars', 'has_url', 'has_numbers']
    metric_counts = [df[m].sum() for m in quality_metrics]
    metric_labels = ['Empty', 'Short (<5)', 'Special Chars', 'Has URL', 'Has Numbers']

    axes[0, 0].bar(metric_labels, metric_counts, color='steelblue')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Text Quality Issues')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 大写比例分布
    axes[0, 1].hist(df['uppercase_ratio'], bins=50, color='coral', edgecolor='white', alpha=0.7)
    axes[0, 1].axvline(df['uppercase_ratio'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["uppercase_ratio"].mean():.3f}')
    axes[0, 1].set_xlabel('Uppercase Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Uppercase Character Ratio Distribution')
    axes[0, 1].legend()

    # 平均词长分布
    axes[1, 0].hist(df['avg_word_length'], bins=40, color='green', edgecolor='white', alpha=0.7)
    axes[1, 0].axvline(df['avg_word_length'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["avg_word_length"].mean():.2f}')
    axes[1, 0].set_xlabel('Average Word Length')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Average Word Length Distribution')
    axes[1, 0].legend()

    # 按标签分组的质量对比
    labels = df[label_col].unique()
    x = np.arange(len(quality_metrics))
    width = 0.35
    colors = ['#2ecc71', '#e74c3c']

    for i, label in enumerate(labels[:2]):
        subset = df[df[label_col] == label]
        values = [subset[m].mean() * 100 for m in quality_metrics]
        axes[1, 1].bar(x + i*width, values, width, label=str(label), color=colors[i])

    axes[1, 1].set_xticks(x + width/2)
    axes[1, 1].set_xticklabels(metric_labels, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_title('Quality Issues by Label')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/text_quality.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "文本质量分析")


def generate_summary(df, label_col, text_col):
    """生成EDA总结 - 不输出到报告"""
    # 仅保存到文件供参考，不添加到HTML报告
    summary = f"""
数据集基本情况
--------------
样本数: {len(df)}
特征数: {df.shape[1]}
缺失值: {df.isnull().sum().sum()}
重复行: {df.duplicated().sum()}

标签分布:
{df[label_col].value_counts().to_string()}

文本统计:
平均长度: {df['text_length'].mean():.2f} 字符
平均词数: {df['word_count'].mean():.2f}
"""
    with open(f'{OUTPUT_DIR}/eda_summary.txt', 'w') as f:
        f.write(summary)


def main():
    """主函数"""
    print("Fake News Detection Dataset - EDA")
    print("=" * 60)

    # 数据路径 - 请修改为你的实际路径
    DATA_PATH = '../data/fakenews 2.csv'

    # 检查文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] 数据文件不存在: {DATA_PATH}")
        print("请从Kaggle下载数据集并放到指定路径:")
        print("https://www.kaggle.com/datasets/iamrahulthorat/fakenews-csv")
        return

    df = load_data(DATA_PATH)

    # 执行EDA步骤
    df = basic_info(df)
    label_col = label_distribution(df)
    text_col = text_analysis(df)
    text_by_label(df, text_col, label_col)
    word_frequency_analysis(df, text_col, label_col)
    ngram_analysis(df, text_col, label_col)
    sample_texts_display(df, text_col, label_col)
    wordcloud_analysis(df, text_col, label_col)
    text_quality_analysis(df, text_col, label_col)
    generate_summary(df, label_col, text_col)

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
