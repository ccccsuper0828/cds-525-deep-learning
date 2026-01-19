"""
Social Media Sentiment Analysis Dataset - Exploratory Data Analysis
数据集: https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis
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

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = '../reports/figures/social_media_sentiment'
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
            <p>Social Media Sentiment Analysis - EDA Report</p>
        </footer>
    </div>
</body>
</html>'''
        return html


# 全局报告对象
report = HTMLReport("Social Media Sentiment Analysis Dataset EDA")


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
    output.write(f"\n数据预览:\n{df.head(10)}\n")

    report.add_section("2. 基本信息统计", output.getvalue())
    return df


def find_columns(df):
    """自动识别文本列和标签列"""
    text_col = None
    label_col = None

    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['text', 'content', 'post', 'tweet', 'message']):
            text_col = col
        if any(keyword in col_lower for keyword in ['sentiment', 'label', 'class', 'target']):
            label_col = col

    if text_col is None:
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.len().mean() > 20:
                text_col = col
                break

    if label_col is None:
        for col in df.columns:
            if col != text_col and df[col].nunique() < 10:
                label_col = col
                break

    print(f"识别到的文本列: {text_col}")
    print(f"识别到的标签列: {label_col}")

    return text_col, label_col


def sentiment_distribution(df, label_col):
    """情感标签分布分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("3. 情感标签分布分析\n")
    output.write("=" * 60 + "\n")

    sentiment_counts = df[label_col].value_counts()
    output.write(f"\n情感分布:\n{sentiment_counts}\n")
    output.write(f"\n情感比例:\n")
    for label, count in sentiment_counts.items():
        output.write(f"  {label}: {count/len(df)*100:.2f}%\n")

    report.add_section("3. 情感标签分布分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(sentiment_counts)))

    # 柱状图
    bars = axes[0].bar(sentiment_counts.index.astype(str), sentiment_counts.values, color=colors)
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Sentiment Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    for bar, count in zip(bars, sentiment_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom', fontsize=10)

    # 饼图
    axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index.astype(str),
               autopct='%1.1f%%', colors=colors, explode=[0.02]*len(sentiment_counts))
    axes[1].set_title('Sentiment Distribution (Pie)')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sentiment_distribution.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "情感分布可视化")


def text_analysis(df, text_col, label_col):
    """文本特征分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("4. 文本特征分析\n")
    output.write("=" * 60 + "\n")

    df['text_length'] = df[text_col].astype(str).apply(len)
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))

    # 社交媒体特有特征
    df['hashtag_count'] = df[text_col].astype(str).apply(lambda x: len(re.findall(r'#\w+', x)))
    df['mention_count'] = df[text_col].astype(str).apply(lambda x: len(re.findall(r'@\w+', x)))
    df['url_count'] = df[text_col].astype(str).apply(lambda x: len(re.findall(r'http\S+', x)))
    df['emoji_count'] = df[text_col].astype(str).apply(
        lambda x: len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', x))
    )

    output.write(f"\n文本长度统计:\n")
    output.write(f"  平均字符数: {df['text_length'].mean():.2f}\n")
    output.write(f"  最大字符数: {df['text_length'].max()}\n")
    output.write(f"  最小字符数: {df['text_length'].min()}\n")

    output.write(f"\n词数统计:\n")
    output.write(f"  平均词数: {df['word_count'].mean():.2f}\n")

    output.write(f"\n社交媒体特征:\n")
    output.write(f"  平均hashtag数: {df['hashtag_count'].mean():.2f}\n")
    output.write(f"  平均@mention数: {df['mention_count'].mean():.2f}\n")
    output.write(f"  平均URL数: {df['url_count'].mean():.2f}\n")

    report.add_section("4. 文本特征分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 文本长度分布
    axes[0, 0].hist(df['text_length'], bins=40, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0, 0].axvline(df['text_length'].mean(), color='red', linestyle='--')
    axes[0, 0].set_xlabel('Character Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Text Length Distribution')

    # 词数分布
    axes[0, 1].hist(df['word_count'], bins=30, color='coral', edgecolor='white', alpha=0.7)
    axes[0, 1].axvline(df['word_count'].mean(), color='red', linestyle='--')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Word Count Distribution')

    # 按情感分组的文本长度
    sentiments = df[label_col].unique()
    for sentiment in sentiments:
        subset = df[df[label_col] == sentiment]
        axes[0, 2].hist(subset['text_length'], bins=30, alpha=0.5, label=str(sentiment))
    axes[0, 2].set_xlabel('Character Count')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Text Length by Sentiment')
    axes[0, 2].legend()

    # 社交媒体特征统计
    social_features = ['hashtag_count', 'mention_count', 'url_count']
    feature_means = [df[f].mean() for f in social_features]
    axes[1, 0].bar(social_features, feature_means, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1, 0].set_ylabel('Average Count')
    axes[1, 0].set_title('Average Social Media Features')
    axes[1, 0].tick_params(axis='x', rotation=15)

    # 按情感的社交特征
    x = np.arange(len(sentiments))
    width = 0.25
    for i, feature in enumerate(social_features):
        values = [df[df[label_col] == s][feature].mean() for s in sentiments]
        axes[1, 1].bar(x + i*width, values, width, label=feature)
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels([str(s) for s in sentiments], rotation=45)
    axes[1, 1].set_ylabel('Average Count')
    axes[1, 1].set_title('Social Features by Sentiment')
    axes[1, 1].legend()

    # 相关性热力图
    corr_cols = ['text_length', 'word_count', 'hashtag_count', 'mention_count']
    corr_matrix = df[corr_cols].corr()
    im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 2].set_xticks(range(len(corr_cols)))
    axes[1, 2].set_yticks(range(len(corr_cols)))
    axes[1, 2].set_xticklabels([c.replace('_', '\n') for c in corr_cols], fontsize=9)
    axes[1, 2].set_yticklabels([c.replace('_', '\n') for c in corr_cols], fontsize=9)
    axes[1, 2].set_title('Feature Correlation')
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/text_analysis.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "文本特征分析")

    return df


def word_frequency_analysis(df, text_col, label_col, top_n=15):
    """词频分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("5. 词频分析\n")
    output.write("=" * 60 + "\n")

    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                'have', 'has', 'had', 'its', 'it', 'as', 'that', 'this', 'will', 'i',
                'you', 'he', 'she', 'we', 'they', 'my', 'your', 'me', 'im', 'just',
                'so', 'do', 'if', 'not', 'no', 'all', 'can', 'get', 'got', 'like',
                'dont', 'what', 'when', 'how', 'rt', 'amp', 'http', 'https', 'co'}

    def tokenize(text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return [w for w in words if w not in stopwords and len(w) > 2]

    sentiments = df[label_col].unique()

    for sentiment in sentiments:
        subset = df[df[label_col] == sentiment]
        all_words = []
        for text in subset[text_col]:
            all_words.extend(tokenize(text))

        word_freq = Counter(all_words).most_common(top_n)

        output.write(f"\nTop {top_n} words for '{sentiment}':\n")
        for word, count in word_freq[:10]:
            output.write(f"  {word}: {count}\n")

    report.add_section("5. 词频分析", output.getvalue())

    # 可视化
    num_sentiments = len(sentiments)
    fig, axes = plt.subplots(1, min(num_sentiments, 4), figsize=(5*min(num_sentiments, 4), 6))
    if num_sentiments == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, num_sentiments))

    for idx, sentiment in enumerate(sentiments[:4]):
        subset = df[df[label_col] == sentiment]
        all_words = []
        for text in subset[text_col]:
            all_words.extend(tokenize(text))

        word_freq = Counter(all_words).most_common(top_n)

        if word_freq:
            words, counts = zip(*word_freq)
            axes[idx].barh(range(len(words)), counts, color=colors[idx])
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Frequency')
            axes[idx].set_title(f'Top Words - {sentiment}')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/word_frequency.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "词频分析")


def sample_texts(df, text_col, label_col, n=3):
    """展示各类别样本文本"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("6. 样本文本展示\n")
    output.write("=" * 60 + "\n")

    for sentiment in df[label_col].unique():
        output.write(f"\n--- {sentiment} ---\n")
        samples = df[df[label_col] == sentiment][text_col].head(n)
        for i, text in enumerate(samples, 1):
            output.write(f"  [{i}] {str(text)[:100]}...\n")

    report.add_section("6. 样本文本展示", output.getvalue())


def wordcloud_analysis(df, text_col, label_col):
    """词云可视化"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("7. 词云分析\n")
    output.write("=" * 60 + "\n")

    try:
        from wordcloud import WordCloud
    except ImportError:
        output.write("WordCloud未安装，跳过\n")
        report.add_section("7. 词云分析", output.getvalue())
        return

    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                'have', 'has', 'had', 'it', 'its', 'this', 'that', 'i', 'you', 'my',
                'im', 'dont', 'just', 'so', 'http', 'https', 'rt', 'amp', 'co'}

    labels = df[label_col].unique()
    n_labels = min(len(labels), 4)
    fig, axes = plt.subplots(1, n_labels, figsize=(5*n_labels, 4))
    if n_labels == 1:
        axes = [axes]

    for idx, label in enumerate(labels[:n_labels]):
        text = ' '.join(df[df[label_col] == label][text_col].astype(str).tolist())
        # 清理社交媒体特殊内容
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        wc = WordCloud(width=500, height=350, background_color='white',
                      stopwords=stopwords, max_words=60)
        wc.generate(text)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'{label}', fontsize=11)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/wordcloud.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "词云分析")


def emoji_sentiment_analysis(df, text_col, label_col):
    """Emoji情感映射分析"""
    output = StringIO()
    output.write("=" * 60 + "\n")
    output.write("8. Emoji情感映射分析\n")
    output.write("=" * 60 + "\n")

    # 定义emoji情感映射
    emoji_sentiment_map = {
        'positive': [
            '\U0001F600', '\U0001F601', '\U0001F602', '\U0001F603', '\U0001F604',  # 笑脸
            '\U0001F60A', '\U0001F60D', '\U0001F618', '\U0001F60E', '\U0001F607',  # 开心
            '\U0001F970', '\U0001F60B', '\U0001F917', '\U0001F929', '\U0001F973',  # 喜爱
            '\U0001F44D', '\U0001F44F', '\U0001F64C', '\U00002764', '\U0001F499',  # 点赞/爱心
            '\U0001F389', '\U0001F38A', '\U0001F381', '\U0001F31F', '\U00002B50',  # 庆祝/星星
            '\U0001F4AA', '\U0001F525', '\U0001F4AF', '\U00002705', '\U0001F3C6'   # 力量/胜利
        ],
        'negative': [
            '\U0001F622', '\U0001F62D', '\U0001F61E', '\U0001F614', '\U0001F629',  # 哭泣
            '\U0001F620', '\U0001F621', '\U0001F624', '\U0001F92C', '\U0001F47F',  # 愤怒
            '\U0001F625', '\U0001F627', '\U0001F628', '\U0001F630', '\U0001F631',  # 担忧/害怕
            '\U0001F4A9', '\U0001F44E', '\U0001F645', '\U0000274C', '\U0001F6AB',  # 不喜欢
            '\U0001F494', '\U0001F915', '\U0001F922', '\U0001F92E', '\U0001F635'   # 心碎/不适
        ],
        'neutral': [
            '\U0001F610', '\U0001F611', '\U0001F914', '\U0001F928', '\U0001F644',  # 中性表情
            '\U0001F60F', '\U0001F612', '\U0001F636', '\U0001F615', '\U0001F61F',  # 思考
            '\U0001F4AC', '\U0001F4AD', '\U0001F4E2', '\U0001F4E3', '\U0001F4F1'   # 对话/通知
        ]
    }

    # 提取所有emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情
        "\U0001F300-\U0001F5FF"  # 符号和象形文字
        "\U0001F680-\U0001F6FF"  # 交通和地图符号
        "\U0001F700-\U0001F77F"  # 炼金术符号
        "\U0001F780-\U0001F7FF"  # 几何形状
        "\U0001F800-\U0001F8FF"  # 补充箭头
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字
        "\U0001FA00-\U0001FA6F"  # 象棋符号
        "\U0001FA70-\U0001FAFF"  # 符号和象形文字扩展
        "\U00002702-\U000027B0"  # 丁贝符号
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    def extract_emojis(text):
        return emoji_pattern.findall(str(text))

    def get_emoji_list(text):
        emojis = extract_emojis(text)
        return list(''.join(emojis))

    def classify_emoji_sentiment(emoji):
        for sentiment, emojis in emoji_sentiment_map.items():
            if emoji in emojis:
                return sentiment
        return 'unknown'

    # 提取emoji并分类
    df['emojis'] = df[text_col].apply(get_emoji_list)
    df['emoji_count_extracted'] = df['emojis'].apply(len)

    # 统计emoji情感
    df['positive_emojis'] = df['emojis'].apply(
        lambda x: sum(1 for e in x if classify_emoji_sentiment(e) == 'positive')
    )
    df['negative_emojis'] = df['emojis'].apply(
        lambda x: sum(1 for e in x if classify_emoji_sentiment(e) == 'negative')
    )
    df['neutral_emojis'] = df['emojis'].apply(
        lambda x: sum(1 for e in x if classify_emoji_sentiment(e) == 'neutral')
    )

    # 收集所有emoji
    all_emojis = []
    for emoji_list in df['emojis']:
        all_emojis.extend(emoji_list)

    emoji_freq = Counter(all_emojis).most_common(20)

    output.write("\nEmoji使用统计:\n")
    output.write(f"  含Emoji的文本数: {(df['emoji_count_extracted'] > 0).sum()} ({(df['emoji_count_extracted'] > 0).mean()*100:.1f}%)\n")
    output.write(f"  平均Emoji数量: {df['emoji_count_extracted'].mean():.2f}\n")
    output.write(f"  最大Emoji数量: {df['emoji_count_extracted'].max()}\n")
    output.write(f"  总Emoji数量: {sum(df['emoji_count_extracted'])}\n")

    output.write(f"\nEmoji情感分类统计:\n")
    output.write(f"  正面Emoji总数: {df['positive_emojis'].sum()}\n")
    output.write(f"  负面Emoji总数: {df['negative_emojis'].sum()}\n")
    output.write(f"  中性Emoji总数: {df['neutral_emojis'].sum()}\n")

    output.write(f"\n最常用Emoji (Top 20):\n")
    for emoji, count in emoji_freq[:20]:
        sentiment = classify_emoji_sentiment(emoji)
        output.write(f"  {emoji}: {count} ({sentiment})\n")

    report.add_section("7. Emoji情感映射分析", output.getvalue())

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Emoji使用频率分布
    axes[0, 0].hist(df['emoji_count_extracted'], bins=20, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0, 0].axvline(df['emoji_count_extracted'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["emoji_count_extracted"].mean():.2f}')
    axes[0, 0].set_xlabel('Emoji Count per Text')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Emoji Usage')
    axes[0, 0].legend()

    # Emoji情感分布
    emoji_sentiments = ['Positive', 'Negative', 'Neutral']
    emoji_totals = [df['positive_emojis'].sum(), df['negative_emojis'].sum(), df['neutral_emojis'].sum()]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    axes[0, 1].bar(emoji_sentiments, emoji_totals, color=colors)
    axes[0, 1].set_ylabel('Total Count')
    axes[0, 1].set_title('Emoji Sentiment Distribution')
    for i, v in enumerate(emoji_totals):
        axes[0, 1].text(i, v + 10, str(v), ha='center')

    # 按文本标签分组的Emoji使用
    sentiments = df[label_col].unique()
    x = np.arange(len(emoji_sentiments))
    width = 0.25

    for i, sentiment in enumerate(sentiments[:3]):
        subset = df[df[label_col] == sentiment]
        values = [
            subset['positive_emojis'].mean(),
            subset['negative_emojis'].mean(),
            subset['neutral_emojis'].mean()
        ]
        offset = (i - 1) * width
        axes[1, 0].bar(x + offset, values, width, label=str(sentiment))

    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(emoji_sentiments)
    axes[1, 0].set_ylabel('Average Count')
    axes[1, 0].set_title('Average Emoji Sentiment by Text Label')
    axes[1, 0].legend()

    # Emoji与标签一致性分析
    def get_emoji_dominant_sentiment(row):
        if row['positive_emojis'] > row['negative_emojis'] and row['positive_emojis'] > row['neutral_emojis']:
            return 'positive'
        elif row['negative_emojis'] > row['positive_emojis'] and row['negative_emojis'] > row['neutral_emojis']:
            return 'negative'
        elif row['neutral_emojis'] > 0:
            return 'neutral'
        return 'none'

    df['emoji_sentiment'] = df.apply(get_emoji_dominant_sentiment, axis=1)

    # 统计emoji情感与文本标签的一致性
    consistency_data = []
    for sentiment in sentiments:
        subset = df[(df[label_col] == sentiment) & (df['emoji_count_extracted'] > 0)]
        if len(subset) > 0:
            consistent = (subset['emoji_sentiment'] == sentiment).mean() * 100
            consistency_data.append({'label': sentiment, 'consistency': consistent, 'count': len(subset)})

    if consistency_data:
        labels = [d['label'] for d in consistency_data]
        consistencies = [d['consistency'] for d in consistency_data]
        axes[1, 1].bar(labels, consistencies, color=plt.cm.Set2(np.linspace(0, 1, len(labels))))
        axes[1, 1].set_ylabel('Consistency (%)')
        axes[1, 1].set_xlabel('Text Label')
        axes[1, 1].set_title('Emoji-Label Sentiment Consistency')
        axes[1, 1].set_ylim(0, 100)
    else:
        axes[1, 1].text(0.5, 0.5, 'No emoji data available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Emoji-Label Sentiment Consistency')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/emoji_sentiment.png', dpi=150, bbox_inches='tight')
    report.add_figure(fig, "Emoji情感分析")


def generate_summary(df, text_col, label_col):
    """保存数据统计"""
    sentiment_counts = df[label_col].value_counts()

    summary = f"""
数据概览
--------
样本: {len(df)}
情感类别数: {df[label_col].nunique()}

分布:
{sentiment_counts.to_string()}

文本长度: {df['text_length'].mean():.2f} (avg)
hashtag: {df['hashtag_count'].mean():.2f} (avg)
mention: {df['mention_count'].mean():.2f} (avg)
"""
    with open(f'{OUTPUT_DIR}/eda_summary.txt', 'w') as f:
        f.write(summary)


def main():
    """主函数"""
    print("Social Media Sentiment Analysis Dataset - EDA")
    print("=" * 60)

    DATA_PATH = '../data/sentiment_analysis 2.csv'

    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] 数据文件不存在: {DATA_PATH}")
        print("请从Kaggle下载数据集:")
        print("https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis")
        return
    
    df = load_data(DATA_PATH)
    text_col, label_col = find_columns(df)

    df = basic_info(df)
    sentiment_distribution(df, label_col)
    df = text_analysis(df, text_col, label_col)
    word_frequency_analysis(df, text_col, label_col)
    sample_texts(df, text_col, label_col)
    wordcloud_analysis(df, text_col, label_col)
    # emoji_sentiment_analysis 已移除 - 该数据集不包含emoji
    generate_summary(df, text_col, label_col)

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
