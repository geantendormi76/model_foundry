# blueprints/text_classification/config.py
from pathlib import Path

# --- 基础路径配置 ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

# --- 蓝图任务定义 ---
TASKS = {
    "is_question": {
        "constitution_func_name": "IS_QUESTION_CONSTITUTION",
        "valid_labels": {"Question", "Statement"},
        "expected_keys": {"text", "label"}, # <--- 为旧蓝图添加期望的数据键
    },
    "confirmation": {
        "constitution_func_name": "CONFIRMATION_CONSTITUTION",
        "valid_labels": {"Affirm", "Deny"},
        "expected_keys": {"text", "label"}, # <--- 为旧蓝图添加期望的数据键
    }
}

# --- 引擎各阶段详细配置 ---
GENERATOR_CONFIG = {
    "MODEL_NAME": "gemini-2.0-flash",
    "TARGET_SAMPLES_PER_TASK": 500,
    "SAMPLES_PER_REQUEST": 25,
}
REFINER_CONFIG = {
    "DEDUPE_THRESHOLD": 0.85,
    "DEDUPE_NUM_PERM": 128,
    "DIVERSITY_MODEL": 'distiluse-base-multilingual-cased-v1',
    "DIVERSITY_SAMPLE_SIZE": 200,
    "TEST_SET_SIZE": 0.2,
    "RANDOM_STATE": 42,
}
TRAINER_CONFIG = {
    "RANDOM_STATE": 42,
}