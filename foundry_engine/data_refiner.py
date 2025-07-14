# foundry_engine/data_refiner.py (V4 - 最终修正版)

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
from typing import Type, Dict, Set, List, Tuple

# --- NER专属辅助函数 ---
def _normalize_ner_data(df: pd.DataFrame) -> pd.DataFrame:
    """强制将NER数据中的tokens规范化为单字，并智能修正tags。"""
    print("  - 阶段B: [NER专属] 规范化数据，强制单字分词...")
    fixed_rows = []
    for _, row in df.iterrows():
        new_tokens, new_tags = [], []
        # 增加对数据类型的校验，以应对API可能返回的非列表格式
        if not isinstance(row.get('tokens'), list) or not isinstance(row.get('tags'), list):
            continue # 跳过格式错误的行
        for token, tag in zip(row['tokens'], row['tags']):
            if len(token) == 1:
                new_tokens.append(token)
                new_tags.append(tag)
            else:
                new_tokens.append(token[0])
                new_tags.append(tag)
                for char in token[1:]:
                    new_tokens.append(char)
                    if tag.startswith('B-'):
                        new_tags.append('I-' + tag[2:])
                    else:
                        new_tags.append(tag)
        fixed_rows.append({'tokens': new_tokens, 'tags': new_tags})
    print(f"    - 规范化完成。")
    return pd.DataFrame(fixed_rows)

# --- 分类任务专属辅助函数 ---
def _clean_and_normalize_classification(df: pd.DataFrame, valid_labels: Set) -> pd.DataFrame:
    """为分类任务清洗和规范化数据。"""
    print("  - 阶段B: [分类专属] 清洗与规范化...")
    df['text'] = df['text'].str.lower().str.strip().apply(lambda x: re.sub(r'\s+', ' ', x))
    initial_count = len(df)
    df = df[df['label'].isin(valid_labels) & (df['text'] != '')]
    print(f"    - 过滤无效数据: 移除了 {initial_count - len(df)} 条无效标签或空文本的样本。")
    return df

# --- 通用辅助函数 ---
def _deep_deduplication(df: pd.DataFrame, is_ner: bool) -> pd.DataFrame:
    """对数据进行精确去重。"""
    print("  - 阶段A: 深度去重...")
    initial_count = len(df)
    
    if is_ner:
        subset_col = 'text_str'
        df[subset_col] = df['tokens'].apply(lambda x: "".join(x) if isinstance(x, list) else "")
    else:
        subset_col = 'text'

    df.drop_duplicates(subset=[subset_col], inplace=True, keep='first')
    
    if is_ner:
        df.drop(columns=[subset_col], inplace=True)
        
    deduped_count = len(df)
    print(f"    - 精确去重: 移除了 {initial_count - deduped_count} 条完全重复的样本。")
    return df

def _analyze_and_report(df: pd.DataFrame, task_name: str, report_path: Path, is_ner: bool):
    """生成数据质量报告。"""
    print("  - 阶段C: 多维质量评估...")
    report = [f"# {task_name} 数据质量分析报告", f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    report.append(f"\n## 1. 基础统计\n- 清洗与去重后总样本数: {len(df)}")
    
    if is_ner:
        all_tags = [tag for tags_list in df['tags'] for tag in tags_list]
        tag_counts = pd.Series(all_tags).value_counts()
        report.append("\n## 2. 标签分布 (按Tag)")
        report.append(tag_counts.to_frame().to_markdown())
    else:
        label_counts = df['label'].value_counts()
        report.append("\n## 2. 标签分布 (按样本)")
        report.append(label_counts.to_frame().to_markdown())

    report_path.write_text("\n".join(report), encoding='utf-8')
    print(f"    - 详细质量报告已保存至: {report_path}")

def _balance_and_split(df: pd.DataFrame, config: Dict, is_ner: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """拆分数据集，并为分类任务做平衡处理。"""
    print("  - 阶段D: 数据集平衡与拆分...")
    stratify_col = None
    if not is_ner:
        min_samples = df['label'].value_counts().min()
        print(f"    - [分类专属] 平衡策略: 对多数类进行欠采样，目标样本数: {min_samples}")
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min_samples, random_state=config['RANDOM_STATE']))
        stratify_col = df['label']

    train_df, test_df = train_test_split(df, test_size=config['TEST_SET_SIZE'], random_state=config['RANDOM_STATE'], stratify=stratify_col)
    print(f"    - 拆分完成: 训练集 {len(train_df)} 条, 测试集 {len(test_df)} 条。")
    return train_df, test_df

def _process_pipeline(raw_file_path: Path, task_config: Dict, config_module: Type, is_ner: bool):
    """根据任务类型（是否为NER）执行不同的处理流水线。"""
    task_name = raw_file_path.stem.replace('_dataset', '')
    refiner_config = config_module.REFINER_CONFIG
    
    print(f"\n{'='*25} 开始处理任务: {task_name} {'='*25}")

    if not raw_file_path.exists():
        print(f"[错误] 原始数据文件不存在: {raw_file_path}，跳过此任务。")
        return

    try:
        df = pd.read_json(raw_file_path, lines=True)
    except ValueError as e:
        print(f"[致命错误] 读取JSONL文件失败: {raw_file_path}。可能是由于API返回了不完整的JSON。错误: {e}")
        return
        
    print(f"步骤 1: 从 {raw_file_path.name} 加载了 {len(df)} 行原始数据。")

    df = _deep_deduplication(df, is_ner)

    if is_ner:
        df = _normalize_ner_data(df)
    else:
        df = _clean_and_normalize_classification(df, task_config["valid_labels"])

    if df.empty:
        print("[错误] 处理后无有效数据，任务终止。")
        return

    # ======================================================================
    # === 编译器指示修正 (V4) ===
    # 错误原因: report_path 变量在使用前未定义。
    # 解决方案: 在调用 _analyze_and_report 之前，明确定义 report_path。
    reports_dir = config_module.DATA_DIR / "reports"
    report_path = reports_dir / f"{task_name}_quality_report_{datetime.now().strftime('%Y%m%d')}.md"
    # ======================================================================
    _analyze_and_report(df, task_name, report_path, is_ner)
    
    train_df, test_df = _balance_and_split(df, refiner_config, is_ner)

    print("\n步骤 6: 保存最终处理结果...")
    processed_data_dir = config_module.DATA_DIR / "processed"
    train_output_path = processed_data_dir / f"{task_name}_train.jsonl"
    test_output_path = processed_data_dir / f"{task_name}_test.jsonl"
    
    train_df.to_json(train_output_path, orient='records', lines=True, force_ascii=False)
    test_df.to_json(test_output_path, orient='records', lines=True, force_ascii=False)
    
    print(f"  - 清理后的训练集 -> {train_output_path}")
    print(f"  - 清理后的测试集 -> {test_output_path}")
    print(f"{'='*25} 任务 {task_name} 处理完毕 {'='*25}\n")

def run(config_module: Type):
    """数据精炼引擎的主入口。"""
    data_dir = config_module.DATA_DIR
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "reports").mkdir(parents=True, exist_ok=True)

    is_ner_blueprint = "tag_map" in next(iter(config_module.TASKS.values()))

    for task_name, task_config in config_module.TASKS.items():
        raw_file = data_dir / "raw" / f"{task_name}_dataset.jsonl"
        _process_pipeline(raw_file, task_config, config_module, is_ner=is_ner_blueprint)