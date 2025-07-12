# /home/zhz/zhzai/model_foundry/scripts/2_refine_and_analyze.py (V1 - 数据精炼厂)

import os
import json
import re
import pandas as pd
import numpy as np
import jieba
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer, util

# --- 配置区 ---
# 获取脚本文件自身所在的目录
SCRIPT_DIR = Path(__file__).parent
# 基于脚本目录，构建到项目根目录(model_foundry)的相对路径
BASE_DIR = SCRIPT_DIR.parent

RAW_DATA_DIR = BASE_DIR / "datasets" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "datasets" / "processed"
REPORTS_DIR = BASE_DIR / "datasets" / "reports"

# MinHashLSH 近似去重配置
# 阈值越低，去重越严格（更容易将两条文本视为相似）
DEDUPE_THRESHOLD = 0.85
DEDUPE_NUM_PERM = 128

# 语义多样性分析配置
# 我们使用一个轻量级的、支持中文的SBERT模型
# 首次运行时会自动下载模型
DIVERSITY_MODEL = 'distiluse-base-multilingual-cased-v1'
DIVERSITY_SAMPLE_SIZE = 500 # 对每个类别抽样500条进行分析，以提高速度

# 数据集拆分配置
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42

# --- 核心功能模块 ---

def deep_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据研究报告第一节，执行深度去重，包含精确去重和近似去重。
    """
    print("  - 阶段A: 深度去重...")
    
    # 1. 精确去重
    initial_count = len(df)
    df.drop_duplicates(subset=['text'], inplace=True)
    exact_deduped_count = len(df)
    print(f"    - 精确去重: 移除了 {initial_count - exact_deduped_count} 条完全重复的样本。")

    # 2. 近似去重 (MinHashLSH)
    lsh = MinHashLSH(threshold=DEDUPE_THRESHOLD, num_perm=DEDUPE_NUM_PERM)
    minhashes = {}
    for index, row in df.iterrows():
        minhash = MinHash(num_perm=DEDUPE_NUM_PERM)
        for word in set(row['text'].split()): # 使用词集合进行哈希
            minhash.update(word.encode('utf8'))
        lsh.insert(index, minhash)
        minhashes[index] = minhash

    print(f"    - MinHashLSH: 已为 {len(df)} 条样本创建索引。正在查询近似副本...")
    
    processed_indices = set()
    duplicate_indices = set()
    for index in df.index:
        if index in processed_indices:
            continue
        result = lsh.query(minhashes[index])
        # 将除自身外的所有相似项标记为重复
        result.remove(index)
        duplicate_indices.update(result)
        processed_indices.update(result)
        processed_indices.add(index)

    df_deduped = df.drop(index=list(duplicate_indices))
    near_deduped_count = len(df_deduped)
    print(f"    - 近似去重: 移除了 {exact_deduped_count - near_deduped_count} 条近似重复的样本。")
    
    return df_deduped

def clean_and_normalize(df: pd.DataFrame, valid_labels: set) -> pd.DataFrame:
    """
    根据研究报告第二节，执行清洗与规范化。
    """
    print("  - 阶段B: 清洗与规范化...")
    
    # 1. 规范化文本：转小写，合并空白字符
    df['text'] = df['text'].str.lower().str.strip()
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    # 2. 过滤无效标签和空文本
    initial_count = len(df)
    df = df[df['label'].isin(valid_labels)]
    df = df[df['text'] != '']
    print(f"    - 过滤无效数据: 移除了 {initial_count - len(df)} 条无效标签或空文本的样本。")
    
    return df

def analyze_and_report(df: pd.DataFrame, task_name: str, report_path: Path):
    """
    根据研究报告补充部分，执行多维质量评估并生成报告。
    """
    print("  - 阶段C: 多维质量评估...")
    report = []
    report.append(f"# {task_name} 数据质量分析报告")
    report.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## 1. 基础统计")
    report.append(f"- 清洗与去重后总样本数: {len(df)}")
    
    # 1. 标签分布
    label_counts = df['label'].value_counts()
    report.append("\n## 2. 标签分布")
    report.append(label_counts.to_markdown())

    # 2. 文本长度分析 (使用jieba分词以更准确地计算中文词数)
    df['word_count'] = df['text'].apply(lambda x: len(list(jieba.cut(x))))
    length_stats = df['word_count'].describe()
    report.append("\n## 3. 文本长度分析 (按词)")
    report.append(length_stats.to_frame().to_markdown())

    # 3. 语义多样性分析
    report.append("\n## 4. 语义多样性分析 (抽样)")
    report.append(f"（使用模型: {DIVERSITY_MODEL}，每个类别抽样: {DIVERSITY_SAMPLE_SIZE}）")
    
    try:
        model = SentenceTransformer(DIVERSITY_MODEL)
        diversity_scores = {}
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            sample_size = min(len(label_df), DIVERSITY_SAMPLE_SIZE)
            if sample_size < 2:
                diversity_scores[label] = "样本过少无法计算"
                continue
            
            sample_embeddings = model.encode(label_df['text'].sample(sample_size, random_state=RANDOM_STATE).tolist(), convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(sample_embeddings, sample_embeddings).numpy()
            # 计算上三角矩阵（不包括对角线）的平均值
            avg_similarity = np.mean(cosine_scores[np.triu_indices_from(cosine_scores, k=1)])
            diversity_scores[label] = f"{1 - avg_similarity:.4f} (1 - 平均余弦相似度)"
        
        report.append("\n| 类别 | 多样性得分 (越低越相似) |")
        report.append("|:---|:---:|")
        for label, score in diversity_scores.items():
            report.append(f"| {label} | {score} |")
    except Exception as e:
        report.append(f"\n计算语义多样性时出错: {e}")
        report.append("请确保已安装 sentence-transformers 并且网络连接正常以下载模型。")

    # 保存报告
    report_path.write_text("\n".join(report), encoding='utf-8')
    print(f"    - 详细质量报告已保存至: {report_path}")

def balance_and_split(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    根据研究报告第三节，执行数据集平衡与拆分。
    """
    print("  - 阶段D: 数据集平衡与拆分...")
    
    # 1. 平衡 (欠采样)
    label_counts = df['label'].value_counts()
    min_samples = label_counts.min()
    print(f"    - 平衡策略: 对多数类进行欠采样，目标样本数: {min_samples}")
    balanced_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min_samples, random_state=RANDOM_STATE))

    # 2. 拆分
    train_df, test_df = train_test_split(
        balanced_df,
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_STATE,
        stratify=balanced_df['label']
    )
    print(f"    - 拆分完成: 训练集 {len(train_df)} 条, 测试集 {len(test_df)} 条。")
    return train_df, test_df

def process_pipeline(raw_file_path: Path, valid_labels: set):
    """完整的数据处理流水线"""
    task_name = raw_file_path.stem.replace('_dataset', '')
    print(f"\n{'='*25} 开始处理任务: {task_name} {'='*25}")

    if not raw_file_path.exists():
        print(f"[错误] 原始数据文件不存在: {raw_file_path}，跳过此任务。")
        return

    df = pd.read_json(raw_file_path, lines=True)
    print(f"步骤 1: 从 {raw_file_path.name} 加载了 {len(df)} 行原始数据。")

    print("\n步骤 2: 执行深度去重...")
    df = deep_deduplication(df)

    print("\n步骤 3: 执行清洗与规范化...")
    df = clean_and_normalize(df, valid_labels)

    if df.empty:
        print("[错误] 处理后无有效数据，任务终止。")
        return

    print(f"\n步骤 4: 生成数据质量报告...")
    report_path = REPORTS_DIR / f"{task_name}_quality_report_{datetime.now().strftime('%Y%m%d')}.md"
    analyze_and_report(df, task_name, report_path)

    print("\n步骤 5: 执行数据集平衡与拆分...")
    train_df, test_df = balance_and_split(df)

    print("\n步骤 6: 保存最终处理结果...")
    train_output_path = PROCESSED_DATA_DIR / f"{task_name}_train.jsonl"
    test_output_path = PROCESSED_DATA_DIR / f"{task_name}_test.jsonl"
    
    train_df[['text', 'label']].to_json(train_output_path, orient='records', lines=True, force_ascii=False)
    test_df[['text', 'label']].to_json(test_output_path, orient='records', lines=True, force_ascii=False)
    
    print(f"  - 清理后的训练集 -> {train_output_path}")
    print(f"  - 清理后的测试集 -> {test_output_path}")
    print(f"{'='*25} 任务 {task_name} 处理完毕 {'='*25}\n")

def main():
    """主执行函数"""
    # 使用 parents=True，它会自动创建所有不存在的父目录
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "raw_file": RAW_DATA_DIR / "is_question_dataset.jsonl",
            "valid_labels": {"Question", "Statement"}
        },
        {
            "raw_file": RAW_DATA_DIR / "confirmation_dataset.jsonl",
            "valid_labels": {"Affirm", "Deny"}
        }
    ]
    
    for task in tasks:
        process_pipeline(task["raw_file"], task["valid_labels"])

if __name__ == "__main__":
    main()