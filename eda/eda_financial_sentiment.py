"""
Financial News Sentiment Analysis Dataset - Exploratory Data Analysis
数据集: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
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

# 设置风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 创建输出目录
OUTPUT_DIR = '../reports/figures/financial_sentiment'
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
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
            border-radius: 20px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0, 184, 148, 0.2);
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
            color: #00b894;
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #00b894;
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
            <h1>💹 {self.title}</h1>
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
            <p>Financial News Sentiment Analysis - EDA Report</p>
        </footer>
    </div>
</body>
</html>'''
        return html


# 全局报告对象
report = HTMLReport("Financial News Sentiment Analysis Dataset EDA")


def load_data(filepath):
    """加载数据集"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("1. 数据加载\n")
    output.write("=" * 60 + "\n")

    # 尝试不同的编码和分隔符
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except:
            df = pd.read_csv(filepath, encoding='utf-8', sep=';')

    output.write(f"数据集形状: {df.shape}\n")
    output.write(f"列名: {df.columns.tolist()}\n")
    output.write(f"\n数据类型:\n{df.dtypes}\n")

    # 标准化列名
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    output.write(f"\n标准化后的列名: {df.columns.tolist()}\n")

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
    output.write(f"\n数据预览:\n{df.head(10)}\n")

    report.add_section("2. 基本信息统计", output.getvalue())
    return df


def sentiment_distribution(df):
    """情感标签分布分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("3. 情感标签分布分析\n")
    output.write("=" * 60 + "\n")

    # 找到情感列
    sentiment_col = None
    for col in df.columns:
        if 'sentiment' in col.lower():
            sentiment_col = col
            break

    if sentiment_col is None:
        sentiment_col = df.columns[0]

    output.write(f"使用情感列: {sentiment_col}\n")

    # 标准化情感标签
    df[sentiment_col] = df[sentiment_col].astype(str).str.strip().str.lower()

    sentiment_counts = df[sentiment_col].value_counts()
    output.write(f"\n情感分布:\n{sentiment_counts}\n")
    output.write(f"\n情感比例:\n")
    for label, count in sentiment_counts.items():
        output.write(f"  {label}: {count/len(df)*100:.2f}%\n")

    report.add_section("3. 情感标签分布分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 定义颜色
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
    color_list = [colors.get(label, '#95a5a6') for label in sentiment_counts.index]

    # 柱状图
    ax1 = axes[0]
    bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=color_list)
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Count')
    ax1.set_title('Sentiment Distribution')
    for bar, count in zip(bars, sentiment_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(count), ha='center', va='bottom', fontsize=11)

    # 饼图
    ax2 = axes[1]
    explode = [0.02] * len(sentiment_counts)
    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index,
            autopct='%1.1f%%', colors=color_list, explode=explode,
            shadow=True, startangle=90)
    ax2.set_title('Sentiment Distribution (Pie)')

    # 类别不平衡可视化
    ax3 = axes[2]
    balance_ratio = sentiment_counts.min() / sentiment_counts.max()
    ax3.barh(['Balance Ratio'], [balance_ratio], color='steelblue')
    ax3.axvline(x=0.5, color='red', linestyle='--', label='50% threshold')
    ax3.axvline(x=0.8, color='orange', linestyle='--', label='80% threshold')
    ax3.set_xlim(0, 1)
    ax3.set_xlabel('Ratio (min_class / max_class)')
    ax3.set_title(f'Class Imbalance: {balance_ratio:.2f}')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sentiment_distribution.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "情感分布可视化")

    return sentiment_col


def headline_analysis(df, sentiment_col):
    """新闻标题分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("4. 新闻标题分析\n")
    output.write("=" * 60 + "\n")

    # 找到文本列
    text_col = None
    for col in df.columns:
        if 'headline' in col.lower() or 'news' in col.lower() or 'text' in col.lower():
            text_col = col
            break

    if text_col is None:
        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    output.write(f"使用文本列: {text_col}\n")

    # 计算文本特征
    df['text_length'] = df[text_col].astype(str).apply(len)
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))

    output.write(f"\n标题长度统计:\n")
    output.write(f"  平均字符数: {df['text_length'].mean():.2f}\n")
    output.write(f"  最小字符数: {df['text_length'].min()}\n")
    output.write(f"  最大字符数: {df['text_length'].max()}\n")
    output.write(f"  标准差: {df['text_length'].std():.2f}\n")

    output.write(f"\n标题词数统计:\n")
    output.write(f"  平均词数: {df['word_count'].mean():.2f}\n")
    output.write(f"  最小词数: {df['word_count'].min()}\n")
    output.write(f"  最大词数: {df['word_count'].max()}\n")

    report.add_section("4. 新闻标题分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 标题长度分布
    axes[0, 0].hist(df['text_length'], bins=40, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0, 0].axvline(df['text_length'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["text_length"].mean():.0f}')
    axes[0, 0].set_xlabel('Character Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Headline Length')
    axes[0, 0].legend()

    # 词数分布
    axes[0, 1].hist(df['word_count'], bins=30, color='coral', edgecolor='white', alpha=0.7)
    axes[0, 1].axvline(df['word_count'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["word_count"].mean():.0f}')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Word Count')
    axes[0, 1].legend()

    # 按情感分组的长度
    sentiments = df[sentiment_col].unique()
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}

    for sentiment in sentiments:
        subset = df[df[sentiment_col] == sentiment]
        color = colors.get(sentiment, '#95a5a6')
        axes[1, 0].hist(subset['text_length'], bins=30, alpha=0.5,
                        label=sentiment, color=color)
    axes[1, 0].set_xlabel('Character Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Headline Length by Sentiment')
    axes[1, 0].legend()

    # 箱线图对比
    data = [df[df[sentiment_col] == s]['word_count'] for s in sentiments]
    bp = axes[1, 1].boxplot(data, labels=sentiments, patch_artist=True)
    for patch, sentiment in zip(bp['boxes'], sentiments):
        patch.set_facecolor(colors.get(sentiment, '#95a5a6'))
        patch.set_alpha(0.6)
    axes[1, 1].set_xlabel('Sentiment')
    axes[1, 1].set_ylabel('Word Count')
    axes[1, 1].set_title('Word Count Distribution by Sentiment')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/headline_analysis.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "新闻标题分析")

    return text_col


def financial_keyword_analysis(df, text_col, sentiment_col):
    """金融关键词分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("5. 金融关键词分析\n")
    output.write("=" * 60 + "\n")

    # 定义金融相关关键词
    financial_keywords = {
        'positive': ['profit', 'growth', 'gain', 'rise', 'increase', 'up', 'high',
                    'strong', 'bullish', 'rally', 'surge', 'beat', 'exceed'],
        'negative': ['loss', 'decline', 'fall', 'drop', 'decrease', 'down', 'low',
                    'weak', 'bearish', 'crash', 'plunge', 'miss', 'below'],
        'neutral': ['report', 'announce', 'plan', 'expect', 'forecast', 'estimate',
                   'market', 'stock', 'share', 'company', 'quarter', 'year']
    }

    def count_keywords(text, keywords):
        text = str(text).lower()
        return sum(1 for word in keywords if word in text)

    # 计算关键词出现次数
    for category, keywords in financial_keywords.items():
        df[f'{category}_keywords'] = df[text_col].apply(lambda x: count_keywords(x, keywords))

    # 统计
    output.write("\n各类关键词平均出现次数 (按情感分组):\n")
    for sentiment in df[sentiment_col].unique():
        subset = df[df[sentiment_col] == sentiment]
        output.write(f"\n  {sentiment}:\n")
        for category in financial_keywords.keys():
            avg = subset[f'{category}_keywords'].mean()
            output.write(f"    {category} keywords: {avg:.2f}\n")

    report.add_section("5. 金融关键词分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 关键词热力图
    sentiments = df[sentiment_col].unique()
    categories = list(financial_keywords.keys())

    heatmap_data = []
    for sentiment in sentiments:
        row = []
        for category in categories:
            avg = df[df[sentiment_col] == sentiment][f'{category}_keywords'].mean()
            row.append(avg)
        heatmap_data.append(row)

    im = axes[0].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(range(len(categories)))
    axes[0].set_xticklabels(categories)
    axes[0].set_yticks(range(len(sentiments)))
    axes[0].set_yticklabels(sentiments)
    axes[0].set_title('Average Keyword Count by Sentiment')
    plt.colorbar(im, ax=axes[0])

    # 添加数值标签
    for i in range(len(sentiments)):
        for j in range(len(categories)):
            axes[0].text(j, i, f'{heatmap_data[i][j]:.2f}', ha='center', va='center')

    # 分组柱状图
    x = np.arange(len(sentiments))
    width = 0.25

    for i, category in enumerate(categories):
        values = [df[df[sentiment_col] == s][f'{category}_keywords'].mean() for s in sentiments]
        offset = (i - 1) * width
        axes[1].bar(x + offset, values, width, label=f'{category} keywords')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sentiments)
    axes[1].set_ylabel('Average Keyword Count')
    axes[1].set_title('Keyword Distribution by Sentiment')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/financial_keywords.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "金融关键词分析")


def word_frequency_by_sentiment(df, text_col, sentiment_col, top_n=15):
    """按情感分组的词频分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("6. 按情感分组的词频分析\n")
    output.write("=" * 60 + "\n")

    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                'have', 'has', 'had', 'its', 'it', 'as', 'that', 'this', 'will'}

    def tokenize(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return [w for w in words if w not in stopwords and len(w) > 2]

    for sentiment in df[sentiment_col].unique():
        subset = df[df[sentiment_col] == sentiment]
        all_words = []
        for text in subset[text_col]:
            all_words.extend(tokenize(text))

        word_freq = Counter(all_words).most_common(top_n)

        output.write(f"\nTop {top_n} words for '{sentiment}':\n")
        for word, count in word_freq[:10]:
            output.write(f"  {word}: {count}\n")

    report.add_section("6. 按情感分组的词频分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}

    for idx, sentiment in enumerate(df[sentiment_col].unique()):
        subset = df[df[sentiment_col] == sentiment]
        all_words = []
        for text in subset[text_col]:
            all_words.extend(tokenize(text))

        word_freq = Counter(all_words).most_common(top_n)

        if word_freq:
            words, counts = zip(*word_freq)
            color = colors.get(sentiment, '#95a5a6')
            axes[idx].barh(range(len(words)), counts, color=color)
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Frequency')
            axes[idx].set_title(f'Top Words - {sentiment.capitalize()}')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/word_frequency_by_sentiment.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "按情感分组的词频分析")


def entity_extraction_analysis(df, text_col, sentiment_col):
    """实体提取分析 (公司名称、股票代码等)"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("7. 实体提取分析\n")
    output.write("=" * 60 + "\n")

    # 常见金融实体模式
    # 股票代码模式 (全大写2-5字母)
    ticker_pattern = r'\b[A-Z]{2,5}\b'
    # 百分比
    percent_pattern = r'\d+\.?\d*\s*%'
    # 金额 (美元)
    money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|m|b|M|B))?'
    # 数字 (可能是股价、收益等)
    number_pattern = r'\b\d+\.?\d*\b'

    # 提取实体
    df['ticker_count'] = df[text_col].astype(str).apply(
        lambda x: len(re.findall(ticker_pattern, x))
    )
    df['percent_count'] = df[text_col].astype(str).apply(
        lambda x: len(re.findall(percent_pattern, x))
    )
    df['money_count'] = df[text_col].astype(str).apply(
        lambda x: len(re.findall(money_pattern, x, re.IGNORECASE))
    )
    df['number_count'] = df[text_col].astype(str).apply(
        lambda x: len(re.findall(number_pattern, x))
    )

    # 提取所有可能的股票代码
    all_tickers = []
    for text in df[text_col]:
        tickers = re.findall(ticker_pattern, str(text))
        # 过滤常见非股票代码的大写词
        common_words = {'CEO', 'CFO', 'IPO', 'GDP', 'USA', 'EUR', 'USD', 'THE', 'AND', 'FOR',
                       'INC', 'LTD', 'PLC', 'LLC', 'ETF', 'NYSE', 'NASDAQ', 'SEC'}
        tickers = [t for t in tickers if t not in common_words]
        all_tickers.extend(tickers)

    ticker_freq = Counter(all_tickers).most_common(20)

    output.write("\n实体统计:\n")
    output.write(f"  含股票代码的文本: {(df['ticker_count'] > 0).sum()} ({(df['ticker_count'] > 0).mean()*100:.1f}%)\n")
    output.write(f"  含百分比的文本: {(df['percent_count'] > 0).sum()} ({(df['percent_count'] > 0).mean()*100:.1f}%)\n")
    output.write(f"  含金额的文本: {(df['money_count'] > 0).sum()} ({(df['money_count'] > 0).mean()*100:.1f}%)\n")
    output.write(f"  含数字的文本: {(df['number_count'] > 0).sum()} ({(df['number_count'] > 0).mean()*100:.1f}%)\n")

    output.write(f"\n最常见的可能股票代码 (Top 20):\n")
    for ticker, count in ticker_freq:
        output.write(f"  {ticker}: {count}\n")

    report.add_section("7. 实体提取分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 实体类型统计
    entity_types = ['Ticker Codes', 'Percentages', 'Money Values', 'Numbers']
    entity_counts = [
        (df['ticker_count'] > 0).sum(),
        (df['percent_count'] > 0).sum(),
        (df['money_count'] > 0).sum(),
        (df['number_count'] > 0).sum()
    ]
    colors = plt.cm.Set2(np.linspace(0, 1, 4))
    axes[0, 0].bar(entity_types, entity_counts, color=colors)
    axes[0, 0].set_ylabel('Number of Headlines')
    axes[0, 0].set_title('Headlines Containing Financial Entities')
    axes[0, 0].tick_params(axis='x', rotation=15)

    # 股票代码频率
    if ticker_freq:
        tickers, counts = zip(*ticker_freq[:15])
        axes[0, 1].barh(range(len(tickers)), counts, color='steelblue')
        axes[0, 1].set_yticks(range(len(tickers)))
        axes[0, 1].set_yticklabels(tickers)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_title('Most Common Ticker Codes')

    # 按情感分组的实体统计
    sentiments = df[sentiment_col].unique()
    sentiment_colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}

    x = np.arange(len(entity_types))
    width = 0.25

    for i, sentiment in enumerate(sentiments):
        subset = df[df[sentiment_col] == sentiment]
        values = [
            (subset['ticker_count'] > 0).mean() * 100,
            (subset['percent_count'] > 0).mean() * 100,
            (subset['money_count'] > 0).mean() * 100,
            (subset['number_count'] > 0).mean() * 100
        ]
        offset = (i - 1) * width
        color = sentiment_colors.get(sentiment, '#95a5a6')
        axes[1, 0].bar(x + offset, values, width, label=sentiment, color=color)

    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(entity_types, rotation=15)
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].set_title('Entity Presence by Sentiment')
    axes[1, 0].legend()

    # 数字实体数量分布
    axes[1, 1].hist(df['number_count'], bins=20, color='coral', edgecolor='white', alpha=0.7)
    axes[1, 1].set_xlabel('Number of Numeric Entities')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Numeric Entities per Headline')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/entity_extraction.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "实体提取分析")


def temporal_keyword_analysis(df, text_col, sentiment_col):
    """时间词汇分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("8. 时间词汇分析\n")
    output.write("=" * 60 + "\n")

    # 定义时间相关词汇
    temporal_keywords = {
        'past': ['last', 'previous', 'ago', 'prior', 'past', 'yesterday', 'earlier'],
        'present': ['today', 'now', 'current', 'currently', 'this', 'present'],
        'future': ['next', 'upcoming', 'future', 'tomorrow', 'forecast', 'expected', 'outlook', 'guidance'],
        'period': ['quarter', 'year', 'month', 'week', 'annual', 'quarterly', 'monthly', 'fiscal', 'h1', 'h2', 'q1', 'q2', 'q3', 'q4']
    }

    def count_temporal(text, keywords):
        text = str(text).lower()
        return sum(1 for word in keywords if word in text.split())

    for category, keywords in temporal_keywords.items():
        df[f'temporal_{category}'] = df[text_col].apply(lambda x: count_temporal(x, keywords))

    # 统计
    output.write("\n时间词汇分布:\n")
    for category in temporal_keywords.keys():
        col = f'temporal_{category}'
        output.write(f"\n  {category.capitalize()} 相关词汇:\n")
        output.write(f"    含该类词汇的标题数: {(df[col] > 0).sum()} ({(df[col] > 0).mean()*100:.1f}%)\n")
        output.write(f"    平均出现次数: {df[col].mean():.2f}\n")

    # 按情感分组
    output.write("\n按情感分组的时间词汇使用:\n")
    for sentiment in df[sentiment_col].unique():
        subset = df[df[sentiment_col] == sentiment]
        output.write(f"\n  {sentiment}:\n")
        for category in temporal_keywords.keys():
            col = f'temporal_{category}'
            output.write(f"    {category}: {subset[col].mean():.3f} avg\n")

    report.add_section("8. 时间词汇分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    categories = list(temporal_keywords.keys())
    sentiment_colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}

    # 整体时间词汇分布
    overall_counts = [(df[f'temporal_{cat}'] > 0).sum() for cat in categories]
    axes[0, 0].bar(categories, overall_counts, color=plt.cm.Set2(np.linspace(0, 1, 4)))
    axes[0, 0].set_ylabel('Number of Headlines')
    axes[0, 0].set_title('Headlines with Temporal Keywords')

    # 按情感分组的热力图
    sentiments = df[sentiment_col].unique()
    heatmap_data = []
    for sentiment in sentiments:
        subset = df[df[sentiment_col] == sentiment]
        row = [subset[f'temporal_{cat}'].mean() for cat in categories]
        heatmap_data.append(row)

    im = axes[0, 1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[0, 1].set_xticks(range(len(categories)))
    axes[0, 1].set_yticks(range(len(sentiments)))
    axes[0, 1].set_xticklabels(categories)
    axes[0, 1].set_yticklabels(sentiments)
    axes[0, 1].set_title('Avg Temporal Keywords by Sentiment')
    plt.colorbar(im, ax=axes[0, 1])

    for i in range(len(sentiments)):
        for j in range(len(categories)):
            axes[0, 1].text(j, i, f'{heatmap_data[i][j]:.2f}', ha='center', va='center')

    # 分组柱状图
    x = np.arange(len(categories))
    width = 0.25

    for i, sentiment in enumerate(sentiments):
        subset = df[df[sentiment_col] == sentiment]
        values = [(subset[f'temporal_{cat}'] > 0).mean() * 100 for cat in categories]
        offset = (i - 1) * width
        color = sentiment_colors.get(sentiment, '#95a5a6')
        axes[1, 0].bar(x + offset, values, width, label=sentiment, color=color)

    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].set_title('Temporal Keyword Usage by Sentiment')
    axes[1, 0].legend()

    # 时间焦点分析 (过去 vs 未来)
    df['temporal_focus'] = df.apply(
        lambda row: 'Future' if row['temporal_future'] > row['temporal_past']
                    else ('Past' if row['temporal_past'] > row['temporal_future'] else 'Balanced'),
        axis=1
    )

    focus_counts = df.groupby([sentiment_col, 'temporal_focus']).size().unstack(fill_value=0)
    focus_counts.plot(kind='bar', ax=axes[1, 1], color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1, 1].set_xlabel('Sentiment')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Temporal Focus by Sentiment')
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].legend(title='Focus')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/temporal_analysis.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "时间词汇分析")


def generate_summary(df, sentiment_col, text_col):
    """生成EDA总结"""
    sentiment_counts = df[sentiment_col].value_counts()

    summary = f"""
Financial News Sentiment Analysis Dataset EDA Summary
=====================================================

1. Dataset Overview:
   - Total samples: {len(df)}
   - Features: {df.shape[1]}
   - Missing values: {df.isnull().sum().sum()}
   - Duplicate rows: {df.duplicated().sum()}

2. Sentiment Distribution:
{sentiment_counts.to_string()}

   Class Imbalance Ratio: {sentiment_counts.min() / sentiment_counts.max():.3f}
   (Ratio < 0.5 indicates significant imbalance)

3. Headline Statistics:
   - Average length: {df['text_length'].mean():.2f} characters
   - Average word count: {df['word_count'].mean():.2f} words
   - Length range: {df['text_length'].min()} - {df['text_length'].max()} chars

4. Key Observations:
   - Dataset has 3 sentiment classes: positive, neutral, negative
   - Neutral class is typically the majority (around 59%)
   - Headlines are relatively short (good for BERT models)
   - Clear class imbalance needs to be addressed

5. Recommendations for Modeling:
   - Use class weights or oversampling for imbalance
   - Consider FinBERT (pre-trained on financial text)
   - Headlines are short, so max_length=128 should suffice
   - Use stratified train/test split

6. Suggested Preprocessing:
   - Lowercase text
   - Remove special characters (optional)
   - No stemming needed for transformer models
"""
    report.add_section("9. EDA 总结报告", summary)

    with open(f'{OUTPUT_DIR}/eda_summary.txt', 'w') as f:
        f.write(summary)


def main():
    """主函数"""
    print("Financial News Sentiment Analysis Dataset - EDA")
    print("=" * 60)

    DATA_PATH = '../data/all-data.csv'  # FinancialPhraseBank 文件名

    if not os.path.exists(DATA_PATH):
        print(f"\n[WARNING] 数据文件不存在: {DATA_PATH}")
        print("请从Kaggle下载数据集:")
        print("https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news")

        print("\n创建示例数据用于演示...")
        demo_data = {
            'sentiment': ['positive'] * 280 + ['negative'] * 130 + ['neutral'] * 590,
            'news_headline': [
                'Company reports strong quarterly profits',
                'Stock prices surge after earnings beat',
                'Revenue growth exceeds expectations'
            ] * 93 + [
                'Company reports quarterly profits'
            ] * 4 + [
                'Shares fall after disappointing results',
                'Company misses revenue targets',
                'Stock plunges on weak outlook'
            ] * 43 + [
                'Shares decline'
            ] + [
                'Company announces quarterly results',
                'CEO to present at investor conference',
                'Board approves dividend policy'
            ] * 196 + [
                'Company plans expansion',
                'Market awaits earnings report'
            ]
        }
        df = pd.DataFrame(demo_data)
        print("使用演示数据进行分析...")
    else:
        df = load_data(DATA_PATH)

    df = basic_info(df)
    sentiment_col = sentiment_distribution(df)
    text_col = headline_analysis(df, sentiment_col)
    financial_keyword_analysis(df, text_col, sentiment_col)
    word_frequency_by_sentiment(df, text_col, sentiment_col)
    entity_extraction_analysis(df, text_col, sentiment_col)
    temporal_keyword_analysis(df, text_col, sentiment_col)
    generate_summary(df, sentiment_col, text_col)

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
