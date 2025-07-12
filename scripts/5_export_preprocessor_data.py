# /model_foundry/scripts/5_export_preprocessor_data.py
# 编码: UTF-8
# 功能: 从.joblib预处理器中提取词汇表和IDF权重，并保存为JSON。
# 这是连接Python训练环境和Rust推理环境的关键桥梁。

import joblib
import json
from pathlib import Path
import numpy as np

def export_data(task_name: str, project_root: Path):
    """
    加载指定任务的预处理器，并将其内部的词汇表和IDF权重导出为JSON文件。
    """
    print(f"--- 开始导出 '{task_name}' 的预处理器数据 ---")
    
    models_dir = project_root / "models"
    preprocessor_path = models_dir / f"{task_name}_preprocessor.joblib"
    output_path = models_dir / f"{task_name}_preprocessor_data.json"

    if not preprocessor_path.exists():
        print(f"[错误] 找不到预处理器文件: {preprocessor_path}")
        print("请确保您已经成功运行了 '3_train_and_evaluate_final.py' 脚本。")
        return

    try:
        pipeline = joblib.load(preprocessor_path)
        
        feature_union = pipeline.named_steps['features']
        word_tfidf = feature_union.transformer_list[0][1]
        char_tfidf = feature_union.transformer_list[1][1]

        # 【核心修正】: 在序列化之前，将词汇表字典中的 numpy.int64 值显式转换为 Python 原生的 int 类型。
        # 我们使用字典推导式来高效地完成这个转换。
        word_vocab_serializable = {k: int(v) for k, v in word_tfidf.vocabulary_.items()}
        char_vocab_serializable = {k: int(v) for k, v in char_tfidf.vocabulary_.items()}

        data_to_export = {
            'word_vocabulary': word_vocab_serializable,
            'word_idf': word_tfidf.idf_.tolist(),
            'char_vocabulary': char_vocab_serializable,
            'char_idf': char_tfidf.idf_.tolist(),
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_export, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 成功! 预处理器数据已导出至: {output_path}")
        print(f"    - 词汇表大小 (词): {len(data_to_export['word_vocabulary'])}")
        print(f"    - 词汇表大小 (字符): {len(data_to_export['char_vocabulary'])}")

    except Exception as e:
        print(f"[严重错误] 导出过程中发生错误: {e}")
        print("请检查 '3_train_and_evaluate_final.py' 中的流水线结构是否已更改。")


def main():
    """
    主执行函数，为所有任务执行导出操作。
    """
    project_root = Path(__file__).resolve().parent.parent
    print(f"项目根目录检测为: {project_root}")
    
    tasks = ["is_question", "confirmation"]
    
    for task in tasks:
        export_data(task, project_root)

if __name__ == "__main__":
    main()