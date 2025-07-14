# foundry_engine/exporter.py (V2 - 支持多蓝图)
import joblib
from pathlib import Path
import subprocess
import sys
import json
from typing import Type

def _ensure_proto_code_generated(scripts_dir: Path):
    """确保 preprocessor_pb2.py 存在，如果不存在则生成。"""
    proto_file = scripts_dir / "preprocessor.proto"
    output_file = scripts_dir / "preprocessor_pb2.py"

    if output_file.exists() and output_file.stat().st_mtime > proto_file.stat().st_mtime:
        return # 已存在且是最新，无需重新生成

    print("--- [Exporter] .proto文件已更新或输出文件不存在，重新生成Python代码 ---")
    if not proto_file.exists():
        raise FileNotFoundError(f"找不到.proto文件: {proto_file}")

    try:
        from grpc_tools import protoc
        command = [
            "grpc_tools.protoc",
            f"--proto_path={scripts_dir}",
            f"--python_out={scripts_dir}",
            str(proto_file.name)
        ]
        if protoc.main(command) != 0:
            raise RuntimeError("protoc.main returned a non-zero exit code.")
        print("✅ Protobuf Python代码生成成功。")
    except Exception as e:
        raise SystemExit(f"[致命错误] protoc 模块执行失败！错误: {e}")

def _export_tfidf_data(task_name: str, config_module: Type, pb2_module):
    """为TF-IDF分类任务导出预处理器数据。"""
    print(f"--- [Exporter] 开始导出 '{task_name}' 的TF-IDF预处理器数据 (Protobuf版) ---")
    models_dir = config_module.MODELS_DIR
    preprocessor_path = models_dir / f"{task_name}_preprocessor.joblib"
    output_path = models_dir / f"{task_name}_preprocessor.bin"

    if not preprocessor_path.exists():
        print(f"[错误] 找不到预处理器文件: {preprocessor_path}。请先执行训练步骤。")
        return

    pipeline = joblib.load(preprocessor_path)
    data_to_export = pb2_module.TfidfPreprocessorData()
    
    def get_ordered_tfidf_data(vectorizer, proto_message):
        vocab_list = [""] * len(vectorizer.vocabulary_)
        for word, index in vectorizer.vocabulary_.items():
            vocab_list[index] = word
        proto_message.vocabulary.extend(vocab_list)
        proto_message.idf_weights.extend(vectorizer.idf_.tolist())

    get_ordered_tfidf_data(pipeline.named_steps['features'].transformer_list[0][1], data_to_export.word_features)
    get_ordered_tfidf_data(pipeline.named_steps['features'].transformer_list[1][1], data_to_export.char_features)
    
    with open(output_path, 'wb') as f:
        f.write(data_to_export.SerializeToString())
    print(f"✅ 成功! TF-IDF预处理器数据已导出至: {output_path}")

def _export_ner_data(task_name: str, config_module: Type, pb2_module):
    """为NER序列标注任务导出预处理器数据。"""
    print(f"--- [Exporter] 开始导出 '{task_name}' 的NER预处理器数据 (Protobuf版) ---")
    models_dir = config_module.MODELS_DIR
    # NER的预处理器数据是在训练时动态生成的，我们从JSON文件中读取
    json_path = models_dir / f"{task_name}_preprocessor.json"
    output_path = models_dir / f"{task_name}_preprocessor.bin"

    if not json_path.exists():
        print(f"[错误] 找不到预处理器JSON文件: {json_path}。请先执行训练步骤。")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        preprocessor_data = json.load(f)

    data_to_export = pb2_module.NerPreprocessorData()
    data_to_export.word_to_ix.update(preprocessor_data["word_to_ix"])
    data_to_export.tag_to_ix.update(preprocessor_data["tag_to_ix"])

    with open(output_path, 'wb') as f:
        f.write(data_to_export.SerializeToString())
    print(f"✅ 成功! NER预处理器数据已导出至: {output_path}")


def run(config_module: Type):
    """预处理器导出引擎的主入口。"""
    scripts_dir = config_module.BASE_DIR / "scripts"
    _ensure_proto_code_generated(scripts_dir)
    
    # 动态导入新生成的pb2模块
    sys.path.insert(0, str(scripts_dir))
    import preprocessor_pb2
    sys.path.pop(0)

    is_ner_blueprint = "tag_map" in next(iter(config_module.TASKS.values()))

    for task_name in config_module.TASKS.keys():
        if is_ner_blueprint:
            _export_ner_data(task_name, config_module, preprocessor_pb2)
        else:
            _export_tfidf_data(task_name, config_module, preprocessor_pb2)