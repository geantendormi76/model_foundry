# /home/zhz/zhzai/model_foundry/scripts/3_train_and_evaluate_final.py

import pandas as pd
import joblib
import time
from pathlib import Path
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

# --- 配置区 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "datasets" / "reports"
RANDOM_STATE = 42

def train_and_export_final_model(task_name: str):
    print(f"\n{'='*25} 开始最终培育模型: {task_name} {'='*25}")

    # 1. 加载数据
    train_df = pd.read_json(PROCESSED_DATA_DIR / f"{task_name}_train.jsonl", lines=True)
    test_df = pd.read_json(PROCESSED_DATA_DIR / f"{task_name}_test.jsonl", lines=True)
    X_train, y_train = train_df['text'], train_df['label']
    X_test, y_test = test_df['text'], test_df['label']

    # 2. 定义特征工程流水线
    feature_engineering_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('word_tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=10000, sublinear_tf=True))
        ]))
    ])

    # 3. 【V-Final核心修改】手动分解流程
    print("\n步骤 1: 对训练集和测试集进行特征提取...")
    X_train_features = feature_engineering_pipeline.fit_transform(X_train)
    X_test_features = feature_engineering_pipeline.transform(X_test)
    print(f"  - 特征提取完成。特征维度: {X_train_features.shape[1]}")

    # 4. 训练一个单独的分类器
    print("\n步骤 2: 训练最终的分类器...")
    # 使用我们从GridSearch中找到的最佳参数
    clf = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, class_weight='balanced', C=20)
    clf.fit(X_train_features, y_train)

    # 5. 评估最终模型
    print("\n步骤 3: 评估最终模型性能...")
    y_pred = clf.predict(X_test_features)
    final_report = classification_report(y_test, y_pred, digits=4)
    print("\n--- 最终性能报告 ---")
    print(final_report)
    
    # 6. 【V-Final核心修改】导出预处理器和简单模型
    print("\n步骤 4: 导出预处理器和ONNX模型...")
    
    # 6.1 保存特征提取流水线 (使用joblib)
    preprocessor_path = MODELS_DIR / f"{task_name}_preprocessor.joblib"
    joblib.dump(feature_engineering_pipeline, preprocessor_path)
    print(f"  - 预处理器已保存至: {preprocessor_path}")

    # 6.2 导出简单的分类器为ONNX
    onnx_model_path = MODELS_DIR / f"{task_name}_classifier.onnx"
    # 输入类型现在是浮点数张量，而不是字符串
    initial_type = [('float_input', FloatTensorType([None, X_train_features.shape[1]]))]
    
    try:
        onx = skl2onnx.to_onnx(clf, initial_types=initial_type, target_opset=15)
        with open(onnx_model_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"  - ONNX模型导出成功！已保存至: {onnx_model_path}")
    except Exception as e:
        print(f"[严重错误] ONNX导出失败: {e}")

    print(f"\n{'='*25} 模型 {task_name} 最终培育完毕 {'='*25}")

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    tasks_to_run = ["is_question", "confirmation"]
    for task in tasks_to_run:
        train_and_export_final_model(task)

if __name__ == "__main__":
    main()