# foundry_engine/exporter.py
import joblib
from pathlib import Path
import subprocess
import sys
from typing import Type

def _ensure_proto_code_generated(scripts_dir: Path):
    """确保 preprocessor_pb2.py 存在，如果不存在则生成。"""
    proto_file = scripts_dir / "preprocessor.proto"
    output_file = scripts_dir / "preprocessor_pb2.py"

    if output_file.exists():
        return # 已存在，无需重新生成

    print("--- [Exporter] 正在从 .proto 文件生成 Python 代码 (首次运行) ---")
    if not proto_file.exists():
        print(f"[致命错误] 找不到 preprocessor.proto 文件于: {scripts_dir}")
        raise FileNotFoundError(f"找不到.proto文件: {proto_file}")

    try:
        # 使用 grpcio-tools 提供的 protoc 命令行工具
        from grpc_tools import protoc
        command = [
            "grpc_tools.protoc",
            f"--proto_path={scripts_dir}",
            f"--python_out={scripts_dir}",
            str(proto_file.name)
        ]
        if protoc.main(command) != 0:
            raise RuntimeError("protoc.main returned a non-zero exit code.")
        print("✅ Protobuf Python 代码生成成功。")
    except Exception as e:
        print(f"[致命错误] protoc 模块执行失败！错误: {e}")
        sys.exit(1)

def _export_task_data(task_name: str, config_module: Type):
    """为单个任务导出预处理器数据。"""
    # 动态导入 protobuf 生成的代码
    scripts_dir = config_module.BASE_DIR / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import preprocessor_pb2
    sys.path.pop(0)

    print(f"--- 开始导出 '{task_name}' 的预处理器数据 (Protobuf版) ---")
    
    models_dir = config_module.MODELS_DIR
    preprocessor_path = models_dir / f"{task_name}_preprocessor.joblib"
    output_path = models_dir / f"{task_name}_preprocessor.bin"

    if not preprocessor_path.exists():
        print(f"[错误] 找不到预处理器文件: {preprocessor_path}。请先执行训练步骤。")
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

# --- 主引擎入口函数 ---
def run(config_module: Type):
    """
    预处理器导出引擎的主入口。
    :param config_module: 从蓝图加载的配置模块。
    """
    scripts_dir = config_module.BASE_DIR / "scripts"
    _ensure_proto_code_generated(scripts_dir)

    for task_name in config_module.TASKS.keys():
        _export_task_data(task_name, config_module)