# /model_foundry/scripts/5_export_preprocessor_data.py (V5 - 最终优化版)
# 编码: UTF-8
# 功能: 独立工作，并将动态导入移至函数内部，消除静态分析警告。

import joblib
from pathlib import Path
import subprocess

def generate_proto_code(scripts_dir: Path):
    print("--- 正在从.proto文件生成Python代码 ---")
    proto_file = scripts_dir / "preprocessor.proto"
    
    if not proto_file.exists():
        print(f"[严重错误] 找不到 preprocessor.proto 文件于: {scripts_dir}")
        raise FileNotFoundError(f"找不到.proto文件: {proto_file}")
        
    protoc_executable = scripts_dir / "protoc_new" / "bin" / "protoc"
    command = [
        str(protoc_executable),
        f"--proto_path={scripts_dir}",
        "--python_out=.",
        str(proto_file.name)
    ]
    try:
        subprocess.run(command, check=True, cwd=scripts_dir, capture_output=True, text=True)
        print("✅ Protobuf Python代码生成成功。")
    except subprocess.CalledProcessError as e:
        print("[严重错误] protoc编译失败！")
        print(f"错误信息: {e.stderr}")
        raise

def export_data(task_name: str, models_dir: Path):
    # 【核心优化】将导入语句移动到函数内部。
    # 这确保了只有在 preprocessor_pb2.py 文件确认已生成后，才尝试导入它。
    import preprocessor_pb2

    print(f"--- 开始导出 '{task_name}' 的预处理器数据 (Protobuf版) ---")
    
    preprocessor_path = models_dir / f"{task_name}_preprocessor.joblib"
    output_path = models_dir / f"{task_name}_preprocessor.bin"

    if not preprocessor_path.exists():
        print(f"[错误] 找不到预处理器文件: {preprocessor_path}")
        return

    try:
        pipeline = joblib.load(preprocessor_path)
        
        def get_ordered_tfidf_data(tfidf_vectorizer, proto_message):
            vocab_list = [""] * len(tfidf_vectorizer.vocabulary_)
            for word, index in tfidf_vectorizer.vocabulary_.items():
                vocab_list[index] = word
            proto_message.vocabulary.extend(vocab_list)
            proto_message.idf_weights.extend(tfidf_vectorizer.idf_.tolist())

        feature_union = pipeline.named_steps['features']
        word_tfidf = feature_union.transformer_list[0][1]
        char_tfidf = feature_union.transformer_list[1][1]

        data_to_export = preprocessor_pb2.PreprocessorData()
        get_ordered_tfidf_data(word_tfidf, data_to_export.word_features)
        get_ordered_tfidf_data(char_tfidf, data_to_export.char_features)
        
        with open(output_path, 'wb') as f:
            f.write(data_to_export.SerializeToString())
        print(f"✅ 成功! Protobuf预处理器数据已导出至: {output_path}")

    except Exception as e:
        print(f"[严重错误] 导出过程中发生错误: {e}")

def main():
    scripts_dir = Path(__file__).resolve().parent
    model_foundry_root = scripts_dir.parent
    models_dir = model_foundry_root / "models"
    
    print(f"Python脚本目录: {scripts_dir}")
    print(f"模型文件目录: {models_dir}")
    
    generate_proto_code(scripts_dir)

    tasks = ["is_question", "confirmation"]
    for task in tasks:
        export_data(task, models_dir)

if __name__ == "__main__":
    main()