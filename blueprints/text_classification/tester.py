# blueprints/text_classification/tester.py

import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys
import jieba

# 动态加载Protobuf生成的代码
scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))
import preprocessor_pb2
sys.path.pop(0)

class PythonClassifier:
    """在Python中模拟Rust端TF-IDF分类器行为的类。"""
    def __init__(self, model_path: Path, preprocessor_path: Path):
        self.session = ort.InferenceSession(str(model_path))
        proto_data = preprocessor_pb2.TfidfPreprocessorData()
        with open(preprocessor_path, 'rb') as f:
            proto_data.ParseFromString(f.read())
        
        self.word_vocab_map = {word: i for i, word in enumerate(proto_data.word_features.vocabulary)}
        self.char_vocab_map = {word: i for i, word in enumerate(proto_data.char_features.vocabulary)}
        self.word_idf = proto_data.word_features.idf_weights
        self.char_idf = proto_data.char_features.idf_weights
        self.jieba = jieba.Tokenizer()

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
            tokens = self.jieba.cut(text, HMM=False)
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

        vector = np.zeros(len(vocab_map), dtype=np.float32)
        for term, count in term_counts.items():
            if term in vocab_map:
                index = vocab_map[term]
                vector[index] = float(count) * idf[index]
        
        norm = np.linalg.norm(vector)
        if norm > 0: vector /= norm
        return vector

    def preprocess(self, text: str) -> np.ndarray:
        lowercased_text = text.lower()
        word_vector = self._calculate_tfidf(lowercased_text, self.word_vocab_map, self.word_idf, False)
        char_vector = self._calculate_tfidf(lowercased_text, self.char_vocab_map, self.char_idf, True)
        return np.concatenate([word_vector, char_vector]).astype(np.float32)

    def predict(self, text: str) -> str:
        input_vector = self.preprocess(text)
        input_tensor = np.expand_dims(input_vector, axis=0)
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: input_tensor})
        return result[0][0]

def run(models_dir: Path):
    """文本分类模型的交互式测试入口。"""
    try:
        is_question_classifier = PythonClassifier(models_dir / "is_question_classifier.onnx", models_dir / "is_question_preprocessor.bin")
        confirmation_classifier = PythonClassifier(models_dir / "confirmation_classifier.onnx", models_dir / "confirmation_preprocessor.bin")
    except Exception as e:
        print(f"\n[严重错误] 加载文本分类模型失败: {e}")
        return

    while True:
        print("\n--- 测试文本分类模型 ---")
        print("  1. '是否为问题' 分类器")
        print("  2. '肯定/否定' 分类器")
        print("  q. 返回主菜单")
        choice = input("请选择 [1/2/q]: ")

        if choice == '1':
            while True:
                text = input("IsQuestion > ")
                if text.lower() == 'q': break
                print(f"  模型预测 ->: **{is_question_classifier.predict(text)}**\n")
        elif choice == '2':
            while True:
                text = input("Confirmation > ")
                if text.lower() == 'q': break
                print(f"  模型预测 ->: **{confirmation_classifier.predict(text)}**\n")
        elif choice.lower() == 'q':
            return