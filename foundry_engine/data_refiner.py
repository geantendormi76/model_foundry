# foundry_engine/data_refiner.py

import os
import re
import pandas as pd
import numpy as np
import jieba
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer, util
from typing import Type, Dict, Set

# --- 辅助函数 (保持不变，但设为私有) ---
def _deep_deduplication(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    print("  - 阶段A: 深度去重...")
    initial_count = len(df)
    df.drop_duplicates(subset=['text'], inplace=True)
    exact_deduped_count = len(df)
    print(f"    - 精确去重: 移除了 {initial_count - exact_deduped_count} 条完全重复的样本。")

    lsh = MinHashLSH(threshold=config["DEDUPE_THRESHOLD"], num_perm=config["DEDUPE_NUM_PERM"])
    minhashes = {}
    for index, row in df.iterrows():
        minhash = MinHash(num_perm=config["DEDUPE_NUM_PERM"])
        for word in set(row['text'].split()):
            minhash.update(word.encode('utf8'))
        lsh.insert(index, minhash)
        minhashes[index] = minhash

    print(f"    - MinHashLSH: 已为 {len(df)} 条样本创建索引。正在查询近似副本...")
    processed_indices, duplicate_indices = set(), set()
    for index in df.index:
        if index in processed_indices:
            continue
        result = lsh.query(minhashes[index])
        result.remove(index)
        duplicate_indices.update(result)
        
        # ======================================================================
        # === 编译器指示修正 (V2) ===
        # 错误原因: result 是一个 list, 不能与 set 进行 | (并集)运算。
        # 解决方案: 在运算前，将 list 类型的 result 显式转换为 set。
        processed_indices.update(set(result) | {index})
        # ======================================================================

    df_deduped = df.drop(index=list(duplicate_indices))
    near_deduped_count = len(df_deduped)
    print(f"    - 近似去重: 移除了 {exact_deduped_count - near_deduped_count} 条近似重复的样本。")
    return df_deduped

def _clean_and_normalize(df: pd.DataFrame, valid_labels: Set) -> pd.DataFrame:
    print("  - 阶段B: 清洗与规范化...")
    df['text'] = df['text'].str.lower().str.strip().apply(lambda x: re.sub(r'\s+', ' ', x))
    initial_count = len(df)
    df = df[df['label'].isin(valid_labels) & (df['text'] != '')]
    print(f"    - 过滤无效数据: 移除了 {initial_count - len(df)} 条无效标签或空文本的样本。")
    return df

def _analyze_and_report(df: pd.DataFrame, task_name: str, report_path: Path, config: Dict):
    print("  - 阶段C: 多维质量评估...")
    report = [f"# {task_name} 数据质量分析报告", f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              "\n## 1. 基础统计", f"- 清洗与去重后总样本数: {len(df)}"]
    
    report.extend(["\n## 2. 标签分布", df['label'].value_counts().to_markdown()])
    
    df['word_count'] = df['text'].apply(lambda x: len(list(jieba.cut(x))))
    report.extend(["\n## 3. 文本长度分析 (按词)", df['word_count'].describe().to_frame().to_markdown()])
    
    report.extend(["\n## 4. 语义多样性分析 (抽样)",
                   f"（使用模型: {config['DIVERSITY_MODEL']}，每个类别抽样: {config['DIVERSITY_SAMPLE_SIZE']}）"])
    try:
        model = SentenceTransformer(config['DIVERSITY_MODEL'])
        diversity_scores = {}
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            sample_size = min(len(label_df), config['DIVERSITY_SAMPLE_SIZE'])
            if sample_size < 2:
                diversity_scores[label] = "样本过少无法计算"
                continue
            sample_embeddings = model.encode(label_df['text'].sample(sample_size, random_state=config['RANDOM_STATE']).tolist(), convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(sample_embeddings, sample_embeddings).numpy()
            avg_similarity = np.mean(cosine_scores[np.triu_indices_from(cosine_scores, k=1)])
            diversity_scores[label] = f"{1 - avg_similarity:.4f} (1 - 平均余弦相似度)"
        report.append("\n| 类别 | 多样性得分 (越低越相似) |")
        report.append("|:---|:---:|")
        for label, score in diversity_scores.items():
            report.append(f"| {label} | {score} |")
    except Exception as e:
        report.append(f"\n计算语义多样性时出错: {e}")

    report_path.write_text("\n".join(report), encoding='utf-8')
    print(f"    - 详细质量报告已保存至: {report_path}")

def _balance_and_split(df: pd.DataFrame, config: Dict) -> (pd.DataFrame, pd.DataFrame):
    print("  - 阶段D: 数据集平衡与拆分...")
    min_samples = df['label'].value_counts().min()
    print(f"    - 平衡策略: 对多数类进行欠采样，目标样本数: {min_samples}")
    balanced_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min_samples, random_state=config['RANDOM_STATE']))
    train_df, test_df = train_test_split(balanced_df, test_size=config['TEST_SET_SIZE'], random_state=config['RANDOM_STATE'], stratify=balanced_df['label'])
    print(f"    - 拆分完成: 训练集 {len(train_df)} 条, 测试集 {len(test_df)} 条。")
    return train_df, test_df

def _process_pipeline(raw_file_path: Path, valid_labels: Set, config_module: Type):
    """单个任务的完整处理流水线。"""
    task_name = raw_file_path.stem.replace('_dataset', '')
    refiner_config = config_module.REFINER_CONFIG
    reports_dir = config_module.DATA_DIR / "reports"
    processed_data_dir = config_module.DATA_DIR / "processed"
    
    print(f"\n{'='*25} 开始处理任务: {task_name} {'='*25}")

    if not raw_file_path.exists():
        print(f"[错误] 原始数据文件不存在: {raw_file_path}，跳过此任务。")
        return

    df = pd.read_json(raw_file_path, lines=True)
    print(f"步骤 1: 从 {raw_file_path.name} 加载了 {len(df)} 行原始数据。")

    df = _deep_deduplication(df, refiner_config)
    df = _clean_and_normalize(df, valid_labels)

    if df.empty:
        print("[错误] 处理后无有效数据，任务终止。")
        return

    report_path = reports_dir / f"{task_name}_quality_report_{datetime.now().strftime('%Y%m%d')}.md"
    _analyze_and_report(df, task_name, report_path, refiner_config)
    train_df, test_df = _balance_and_split(df, refiner_config)

    print("\n步骤 6: 保存最终处理结果...")
    train_output_path = processed_data_dir / f"{task_name}_train.jsonl"
    test_output_path = processed_data_dir / f"{task_name}_test.jsonl"
    train_df[['text', 'label']].to_json(train_output_path, orient='records', lines=True, force_ascii=False)
    test_df[['text', 'label']].to_json(test_output_path, orient='records', lines=True, force_ascii=False)
    
    print(f"  - 清理后的训练集 -> {train_output_path}")
    print(f"  - 清理后的测试集 -> {test_output_path}")
    print(f"{'='*25} 任务 {task_name} 处理完毕 {'='*25}\n")

# --- 主引擎入口函数 ---
def run(config_module: Type):
    """
    数据精炼引擎的主入口。
    :param config_module: 从蓝图加载的配置模块。
    """
    data_dir = config_module.DATA_DIR
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "reports").mkdir(parents=True, exist_ok=True)

    for task_name, task_config in config_module.TASKS.items():
        raw_file = data_dir / "raw" / f"{task_name}_dataset.jsonl"
        valid_labels = task_config["valid_labels"]
        _process_pipeline(raw_file, valid_labels, config_module)