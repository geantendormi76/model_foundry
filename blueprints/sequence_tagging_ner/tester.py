# blueprints/sequence_tagging_ner/tester.py

import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys
from typing import List, Tuple

# 动态加载Protobuf生成的代码
scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))
import preprocessor_pb2
sys.path.pop(0)

class NerPredictor:
    """在Python中模拟NER模型行为的类。"""
    def __init__(self, model_path: Path, preprocessor_path: Path):
        self.session = ort.InferenceSession(str(model_path))
        proto_data = preprocessor_pb2.NerPreprocessorData()
        with open(preprocessor_path, 'rb') as f:
            proto_data.ParseFromString(f.read())
        
        self.word_to_ix = dict(proto_data.word_to_ix)
        self.ix_to_tag = {i: t for t, i in proto_data.tag_to_ix.items()}
        if "<UNK>" not in self.word_to_ix:
            self.word_to_ix["<UNK>"] = max(self.word_to_ix.values()) + 1

    def predict(self, text: str) -> List[Tuple[str, str]]:
        tokens = list(text)
        indices = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w in tokens]
        input_tensor = np.array([indices], dtype=np.int64)

        input_name = self.session.get_inputs()[0].name
        emission_scores = self.session.run(None, {input_name: input_tensor})[0]
        tag_indices = np.argmax(emission_scores, axis=2)[0]
        
        results = [(token, self.ix_to_tag.get(tag_ix, "O")) for token, tag_ix in zip(tokens, tag_indices)]
        return results

def run(models_dir: Path):
    """NER模型的交互式测试入口。"""
    try:
        ner_predictor = NerPredictor(models_dir / "ner_core_entity.onnx", models_dir / "ner_core_entity_preprocessor.bin")
    except Exception as e:
        print(f"\n[严重错误] 加载NER模型失败: {e}")
        return
    
    print("\n--- 测试 '核心实体识别' 模型 ---")
    while True:
        text = input("NER > ")
        if text.lower() == 'q': break
        predictions = ner_predictor.predict(text)
        pretty_output = [f"[{tok}|{tag}]" if tag != 'O' else tok for tok, tag in predictions]
        print(f"  模型预测 ->: {' '.join(pretty_output)}\n")