# 编码: UTF-8
# 功能: 加载训练好的.joblib预处理器和.onnx模型，进行端到端的推理验证。
# 这是将模型移植到Rust前的最后一道质量保证关卡。

import joblib
import onnxruntime as ort
from pathlib import Path
import numpy as np

def verify_models():
    """
    主验证函数，加载模型和预处理器，并对预设的测试样本进行推理。
    """
    print("="*80)
    print("ZhzAI '小脑计划' 微模型验证程序启动")
    print("="*80)

    # --- 路径设置 (稳健的路径管理) ---
    try:
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent
    except NameError:
        project_root = Path('.').resolve()

    models_dir = project_root / "models"
    print(f"项目根目录: {project_root}")
    print(f"模型目录: {models_dir}\n")

    # --- 加载模型和预处理器 ---
    try:
        is_question_preprocessor = joblib.load(models_dir / "is_question_preprocessor.joblib")
        is_question_session = ort.InferenceSession(str(models_dir / "is_question_classifier.onnx"))
        print("'is_question' 模型及预处理器加载成功。")

        confirmation_preprocessor = joblib.load(models_dir / "confirmation_preprocessor.joblib")
        confirmation_session = ort.InferenceSession(str(models_dir / "confirmation_classifier.onnx"))
        print("'confirmation' 模型及预处理器加载成功。\n")
    except Exception as e:
        print(f"[错误] 加载模型文件失败: {e}")
        print("请确保'models'文件夹中包含以下所有文件：")
        print("- is_question_preprocessor.joblib")
        print("- is_question_classifier.onnx")
        print("- confirmation_preprocessor.joblib")
        print("- confirmation_classifier.onnx")
        return

    # --- 定义未见过的测试样本 ---
    is_question_samples = [
        "今天天气怎么样",
        "你能帮我做什么？",
        "我刚才说了啥",
        "搜索一下最近的新闻",
        "好的，就这么办",
        "这是一个测试。",
        "为什么天空是蓝色的",
        "介绍一下你自己",
        "how are you",
        "明白了",
        "这是问题吗？",
    ]

    confirmation_samples = [
        "是的",
        "没错",
        "ok",
        "可以",
        "不对",
        "不是这样的",
        "cancel",
        "取消操作",
        "我确认",
        "请继续",
        "别",
        "我不知道",
        "让我想想",
        "今天天气真好",
    ]
    
    # --- 执行推理并打印结果 ---

    # 1. 验证 'is_question' 分类器
    print("\n--- 验证 'is_question' 分类器 ---")
    
    is_question_features = is_question_preprocessor.transform(is_question_samples)
    is_question_input_name = is_question_session.get_inputs()[0].name
    is_question_input = is_question_features.toarray().astype(np.float32)

    # is_question_outputs[0] 是一个包含字符串标签 ("Question", "Statement") 的数组
    is_question_outputs = is_question_session.run(None, {is_question_input_name: is_question_input})
    is_question_labels = is_question_outputs[0]

    # **【核心修正】** 直接使用模型输出的字符串标签进行判断和打印
    for text, label in zip(is_question_samples, is_question_labels):
        # label 现在是 "Question" 或 "Statement" 字符串
        print(f"输入: '{text}' -> 模型预测: '{label}'")

    # 2. 验证 'confirmation' 分类器
    print("\n--- 验证 'confirmation' 分类器 ---")
    confirmation_features = confirmation_preprocessor.transform(confirmation_samples)
    
    confirmation_input_name = confirmation_session.get_inputs()[0].name
    confirmation_input = confirmation_features.toarray().astype(np.float32)
    
    # confirmation_outputs[0] 是一个包含字符串标签 ("Affirm", "Deny") 的数组
    confirmation_outputs = confirmation_session.run(None, {confirmation_input_name: confirmation_input})
    confirmation_labels = confirmation_outputs[0]

    # **【核心修正】** 不再需要 label_map，直接打印模型输出的字符串标签
    for text, label in zip(confirmation_samples, confirmation_labels):
        # label 现在是 "Affirm" 或 "Deny" 字符串
        print(f"输入: '{text}' -> 模型预测: '{label}'")

    print("\n" + "="*80)
    print("验证完成。请检查以上输出是否符合预期。")
    print("="*80)


if __name__ == "__main__":
    verify_models()