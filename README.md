# CDS 525 Group Project - Dataset EDA

## 项目结构

```
CDS525_Project/
├── data/                    # 数据文件目录 (需要从Kaggle下载)
├── eda/                     # EDA脚本
│   ├── eda_fakenews.py
│   ├── eda_financial_sentiment.py
│   ├── eda_social_media_sentiment.py
│   ├── eda_brain_activity.py
│   ├── eda_kmnist.py
│   └── eda_cifar100.py
├── reports/                 # 报告和图表
│   ├── dataset_research_report.md
│   └── figures/            # EDA生成的图表
├── environment.yml          # Conda环境配置
└── README.md
```

## 快速开始

### 1. 创建Conda环境

```bash
# 方法1: 使用environment.yml (推荐)
conda env create -f environment.yml
conda activate cds525

# 方法2: 手动创建
conda create -n cds525 python=3.10 -y
conda activate cds525
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas numpy scikit-learn matplotlib seaborn jupyter -c conda-forge
pip install transformers datasets accelerate wordcloud
```

### 2. 下载数据集

从Kaggle下载对应数据集到 `data/` 目录:

| 数据集 | 下载链接 | 文件名 |
|-------|---------|--------|
| Fake News | [Kaggle](https://www.kaggle.com/datasets/iamrahulthorat/fakenews-csv) | fakenews.csv |
| Financial Sentiment | [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) | all-data.csv |
| Social Media | [Kaggle](https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis) | sentiment_analysis.csv |
| Brain Activity | [Kaggle](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) | (多个文件) |
| KMNIST | PyTorch内置 | 自动下载 |
| CIFAR-100 | PyTorch内置 | 自动下载 |

### 3. 运行EDA

```bash
cd eda

# 运行单个EDA脚本
python eda_kmnist.py
python eda_cifar100.py
python eda_fakenews.py
python eda_financial_sentiment.py
python eda_social_media_sentiment.py
python eda_brain_activity.py
```

