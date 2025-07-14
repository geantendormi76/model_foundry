# foundry_engine/trainers/tfidf_classifier_trainer.py

import pandas as pd
import joblib
from pathlib import Path
from typing import Type

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

def _train_and_export_task_model(task_name: str, config_module: Type):
    """为单个任务训练并导出模型。"""
    print(f"\n{'='*25} 开始培育模型: {task_name} {'='*25}")
    
    processed_data_dir = config_module.DATA_DIR / "processed"
    models_dir = config_module.MODELS_DIR
    random_state = config_module.TRAINER_CONFIG["RANDOM_STATE"]

    # 1. 加载数据
    train_df = pd.read_json(processed_data_dir / f"{task_name}_train.jsonl", lines=True)
    test_df = pd.read_json(processed_data_dir / f"{task_name}_test.jsonl", lines=True)
    X_train, y_train = train_df['text'], train_df['label']
    X_test, y_test = test_df['text'], test_df['label']

    # 2. 定义特征工程流水线
    feature_engineering_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('word_tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=10000, sublinear_tf=True))
        ]))
    ])

    print("\n步骤 1: 对训练集和测试集进行特征提取...")
    X_train_features = feature_engineering_pipeline.fit_transform(X_train)
    X_test_features = feature_engineering_pipeline.transform(X_test)
    print(f"  - 特征提取完成。特征维度: {X_train_features.shape[1]}")

    print("\n步骤 2: 训练最终的分类器...")
    clf = LogisticRegression(solver='liblinear', random_state=random_state, class_weight='balanced', C=20)
    clf.fit(X_train_features, y_train)

    print("\n步骤 3: 评估最终模型性能...")
    y_pred = clf.predict(X_test_features)
    final_report = classification_report(y_test, y_pred, digits=4)
    print("\n--- 最终性能报告 ---")
    print(final_report)
    
    print("\n步骤 4: 导出预处理器和ONNX模型...")
    preprocessor_path = models_dir / f"{task_name}_preprocessor.joblib"
    joblib.dump(feature_engineering_pipeline, preprocessor_path)
    print(f"  - 预处理器已保存至: {preprocessor_path}")

    onnx_model_path = models_dir / f"{task_name}_classifier.onnx"
    initial_type = [('float_input', FloatTensorType([None, X_train_features.shape[1]]))]
    try:
        onx = skl2onnx.to_onnx(clf, initial_types=initial_type, target_opset=15)
        with open(onnx_model_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"  - ONNX模型导出成功！已保存至: {onnx_model_path}")
    except Exception as e:
        print(f"[严重错误] ONNX导出失败: {e}")

    print(f"\n{'='*25} 模型 {task_name} 培育完毕 {'='*25}")

# --- 主引擎入口函数 ---
def run(config_module: Type):
    """
    TF-IDF 分类器训练引擎的主入口。
    :param config_module: 从蓝图加载的配置模块。
    """
    models_dir = config_module.MODELS_DIR
    models_dir.mkdir(exist_ok=True)
    
    for task_name in config_module.TASKS.keys():
        _train_and_export_task_model(task_name, config_module)