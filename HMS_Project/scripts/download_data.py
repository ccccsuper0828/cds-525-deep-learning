#!/usr/bin/env python
"""
HMS Kaggle比赛数据下载脚本

使用方法:
1. 首先配置Kaggle API:
   - 访问 https://www.kaggle.com/settings
   - 点击 "Create New Token" 下载 kaggle.json
   - 将 kaggle.json 放到 ~/.kaggle/ 目录下
   - chmod 600 ~/.kaggle/kaggle.json

2. 运行本脚本:
   python scripts/download_data.py --data_dir ./data

或者使用环境变量:
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_key"
   python scripts/download_data.py
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import zipfile
import shutil


def check_kaggle_api():
    """检查Kaggle API是否配置正确"""
    try:
        import kaggle
        kaggle.api.authenticate()
        print("✓ Kaggle API configured successfully")
        return True
    except Exception as e:
        print(f"✗ Kaggle API not configured: {e}")
        print("\nPlease configure Kaggle API:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' to download kaggle.json")
        print("3. Place kaggle.json in ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False


def download_competition_data(competition: str, data_dir: str):
    """下载比赛数据"""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    os.makedirs(data_dir, exist_ok=True)

    print(f"\nDownloading competition data: {competition}")
    print(f"Target directory: {data_dir}")
    print("This may take a while (data is ~50GB)...\n")

    try:
        api.competition_download_files(
            competition,
            path=data_dir,
            quiet=False
        )
        print("\n✓ Download completed!")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def extract_data(data_dir: str):
    """解压数据文件"""
    print("\nExtracting downloaded files...")

    zip_files = list(Path(data_dir).glob("*.zip"))

    for zip_file in zip_files:
        print(f"  Extracting: {zip_file.name}")
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(data_dir)
            # 删除zip文件以节省空间
            # os.remove(zip_file)
            print(f"  ✓ {zip_file.name} extracted")
        except Exception as e:
            print(f"  ✗ Failed to extract {zip_file.name}: {e}")

    print("\n✓ Extraction completed!")


def verify_data(data_dir: str):
    """验证数据完整性"""
    print("\nVerifying data integrity...")

    required_files = [
        'train.csv',
        'test.csv',
        'sample_submission.csv'
    ]

    required_dirs = [
        'train_eegs',
        'test_eegs',
        'train_spectrograms',
        'test_spectrograms'
    ]

    all_ok = True

    for f in required_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"  ✓ {f} ({size:.1f} MB)")
        else:
            print(f"  ✗ {f} NOT FOUND")
            all_ok = False

    for d in required_dirs:
        path = os.path.join(data_dir, d)
        if os.path.exists(path) and os.path.isdir(path):
            n_files = len(os.listdir(path))
            print(f"  ✓ {d}/ ({n_files} files)")
        else:
            print(f"  ✗ {d}/ NOT FOUND")
            all_ok = False

    if all_ok:
        print("\n✓ All required files present!")
    else:
        print("\n✗ Some files are missing!")

    return all_ok


def update_config(data_dir: str, config_path: str):
    """更新配置文件中的数据路径"""
    import yaml

    abs_data_dir = os.path.abspath(data_dir)

    print(f"\nUpdating config file: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config['data']['train_csv'] = os.path.join(abs_data_dir, 'train.csv')
        config['data']['eeg_dir'] = os.path.join(abs_data_dir, 'train_eegs')
        config['data']['spec_dir'] = os.path.join(abs_data_dir, 'train_spectrograms')

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print("✓ Config file updated with data paths")
    except Exception as e:
        print(f"✗ Failed to update config: {e}")


def main():
    parser = argparse.ArgumentParser(description='Download HMS competition data')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to store downloaded data')
    parser.add_argument('--competition', type=str,
                       default='hms-harmful-brain-activity-classification',
                       help='Kaggle competition name')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip download, only extract and verify')
    parser.add_argument('--update_config', type=str, default=None,
                       help='Path to config.yaml to update')
    args = parser.parse_args()

    print("="*60)
    print("HMS Competition Data Downloader")
    print("="*60)

    # 检查Kaggle API
    if not args.skip_download:
        if not check_kaggle_api():
            sys.exit(1)

        # 下载数据
        success = download_competition_data(args.competition, args.data_dir)
        if not success:
            print("\nTrying alternative download method...")
            # 尝试命令行方式
            cmd = f"kaggle competitions download -c {args.competition} -p {args.data_dir}"
            subprocess.run(cmd, shell=True)

    # 解压数据
    extract_data(args.data_dir)

    # 验证数据
    verify_data(args.data_dir)

    # 更新配置文件
    if args.update_config:
        update_config(args.data_dir, args.update_config)

    print("\n" + "="*60)
    print("Data preparation completed!")
    print("="*60)

    print(f"\nNext steps:")
    print(f"1. Update config.yaml with correct data paths")
    print(f"2. Run training: python scripts/train.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
