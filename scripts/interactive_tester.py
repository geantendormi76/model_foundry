# /model_foundry/scripts/7_test_inference_in_python.py
# 编码: UTF-8
# 功能: 最终验证脚本。加载最终产物(.onnx, .bin)，并在Python中模拟Rust端的推理流程。
# 这是一个交互式的“模型唤醒实验室”，让用户能直观感受自己培育出的模型的威力。

import onnxruntime as ort
import numpy as np
from pathlib import Path
import jieba
import subprocess
import sys
from grpc_tools import protoc

# --- 确保Protobuf代码已生成 (V4 - 路径修正版) ---
def ensure_proto_code_generated():
    # 【核心修正】我们现在直接使用当前脚本所在的目录作为基准路径。
    # __file__ 指向当前脚本的路径。
    script_dir = Path(__file__).resolve().parent
    proto_file = script_dir / "preprocessor.proto"
    
    # 检查标记文件是否存在，避免重复编译
    # 生成的Python文件将与.proto文件位于同一目录
    if (script_dir / "preprocessor_pb2.py").exists():
        return
    
    print("--- 正在从.proto文件生成Python代码 (首次运行) ---")
    
    # 我们不再需要复杂的路径拼接，直接使用script_dir
    command = [
        "grpc_tools.protoc",
        f"--proto_path={script_dir}",
        f"--python_out={script_dir}",
        str(proto_file)
    ]
    
    try:
        if protoc.main(command) != 0:
            raise RuntimeError("protoc.main returned a non-zero exit code.")
        print("✅ Protobuf Python代码生成成功。")
    except Exception as e:
        print(f"[严重错误] protoc模块执行失败！错误: {e}")
        sys.exit(1)

# --- 【核心修正】动态导入生成的代码 ---
# 在所有函数之外，脚本一加载就执行
ensure_proto_code_generated()
# 将脚本目录添加到sys.path，以便Python可以找到新生成的模块
script_dir_str = str(Path(__file__).resolve().parent)
if script_dir_str not in sys.path:
    sys.path.insert(0, script_dir_str)
import preprocessor_pb2

class PythonClassifier:
    """
    一个在Python中模拟Rust端Classifier行为的类，用于最终验证。
    """
    # --- 【核心修正 V3】__init__方法现在能自动定位Jieba的默认词典 ---
    def __init__(self, model_path: Path, preprocessor_path: Path):
        print(f"正在加载模型: {model_path.name}")
        self.session = ort.InferenceSession(str(model_path))
        
        self.preprocessor_data = preprocessor_pb2.PreprocessorData()
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor_data.ParseFromString(f.read())
        
        self.word_vocab_map = {word: i for i, word in enumerate(self.preprocessor_data.word_features.vocabulary)}
        self.char_vocab_map = {word: i for i, word in enumerate(self.preprocessor_data.char_features.vocabulary)}
        
        # --- 智能加载Jieba词典的最终方案 ---
        try:
            # 1. 直接从jieba库获取其默认词典的路径
            default_dict_path = jieba.get_default_dict_file()
            print(f"  成功定位到Jieba内置词典: {default_dict_path}")
            # 2. 使用这个绝对正确的路径来初始化Tokenizer
            self.jieba = jieba.Tokenizer(dictionary=str(default_dict_path))
        except Exception as e:
            # 3. 如果因任何原因失败，则回退到最基础的初始化方式
            print(f"  [警告] 自动定位Jieba词典失败: {e}。将使用Jieba的内存默认词典。")
            self.jieba = jieba.Tokenizer()

        print("模型加载成功！\n")

    def _calculate_tfidf(self, text: str, vocab_map: dict, idf: list, is_char_ngram: bool) -> np.ndarray:
        term_counts = {}
        
        if is_char_ngram:
            chars = list(text)
            for n in range(2, 6):
                if len(chars) >= n:
                    for i in range(len(chars) - n + 1):
                        ngram = "".join(chars[i:i+n])
                        term_counts[ngram] = term_counts.get(ngram, 0) + 1
        else:
            # 禁用HMM以与Rust端保持一致
            tokens = self.jieba.cut(text, HMM=False)
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

        vector = np.zeros(len(vocab_map), dtype=np.float32)
        for term, count in term_counts.items():
            if term in vocab_map:
                index = vocab_map[term]
                tf = float(count)
                vector[index] = tf * idf[index]
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        
        return vector

    def preprocess(self, text: str) -> np.ndarray:
        lowercased_text = text.lower()
        
        word_vector = self._calculate_tfidf(
            lowercased_text, 
            self.word_vocab_map, 
            self.preprocessor_data.word_features.idf_weights, 
            is_char_ngram=False
        )
        char_vector = self._calculate_tfidf(
            lowercased_text, 
            self.char_vocab_map, 
            self.preprocessor_data.char_features.idf_weights, 
            is_char_ngram=True
        )
        
        return np.concatenate([word_vector, char_vector]).astype(np.float32)

    def predict(self, text: str) -> str:
        # 预处理
        input_vector = self.preprocess(text)
        
        # ONNX Runtime需要一个二维的输入
        input_tensor = np.expand_dims(input_vector, axis=0)
        
        # 推理
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        result = self.session.run([output_name], {input_name: input_tensor})
        
        return result[0][0]

def main():
    """
    主执行函数，提供一个交互式的测试环境。
    """
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    
    print("="*50)
    print("  欢迎来到zhzAI“模型唤醒实验室”！")
    print("  我们将加载您亲手铸造的最终模型产物进行测试。")
    print("="*50)

    try:
        # 不再需要手动指定dict_path
        is_question_classifier = PythonClassifier(
            models_dir / "is_question_classifier.onnx",
            models_dir / "is_question_preprocessor.bin"
        )
        confirmation_classifier = PythonClassifier(
            models_dir / "confirmation_classifier.onnx",
            models_dir / "confirmation_preprocessor.bin"
        )
    except Exception as e:
        print(f"\n[严重错误] 加载模型失败: {e}")
        print("请确保您已经成功并完整地运行了1-5号流水线脚本。")
        return

    print("\n--- 任务1: “是否为问题”分类器测试 ---")
    print("请输入任意一句话，看看模型如何判断 (输入'q'退出):")
    while True:
        text = input("> ")
        if text.lower() == 'q':
            break
        label = is_question_classifier.predict(text)
        print(f"  模型预测 ->: **{label}**\n")

    print("\n--- 任务2: “肯定/否定”分类器测试 ---")
    print("现在，模拟AI向您确认，请输入您的回答 (输入'q'退出):")
    while True:
        text = input("> ")
        if text.lower() == 'q':
            break
        label = confirmation_classifier.predict(text)
        print(f"  模型预测 ->: **{label}**\n")
        
    print("\n实验结束！恭喜您，成功唤醒了您亲手铸造的AI大脑！")
    print("现在，尝试修改 `prompt_constitutions.py`，创造属于你自己的微模型吧！")

if __name__ == "__main__":
    main()