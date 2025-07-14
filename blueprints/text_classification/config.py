# blueprints/text_classification/config.py
from pathlib import Path

# --- 基础路径配置 ---
# BASE_DIR 是整个 model_foundry 项目的根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

# --- 蓝图任务定义 ---
# 定义此蓝图下需要处理的所有模型任务
TASKS = {
    "is_question": {
        "constitution_func_name": "IS_QUESTION_CONSTITUTION",
        "valid_labels": {"Question", "Statement"},
    },
    "confirmation": {
        "constitution_func_name": "CONFIRMATION_CONSTITUTION",
        "valid_labels": {"Affirm", "Deny"},
    }
}

# --- 引擎各阶段详细配置 ---

# 1. data_generator (数据生成器) 配置
GENERATOR_CONFIG = {
    "MODEL_NAME": "gemini-1.5-flash", # 使用您可用的模型
    "TARGET_SAMPLES_PER_TASK": 500, # 为演示，减少数量，可随时调回5000
    "SAMPLES_PER_REQUEST": 25,
}

# 2. data_refiner (数据精炼器) 配置
REFINER_CONFIG = {
    "DEDUPE_THRESHOLD": 0.85,
    "DEDUPE_NUM_PERM": 128,
    "DIVERSITY_MODEL": 'distiluse-base-multilingual-cased-v1',
    "DIVERSITY_SAMPLE_SIZE": 200, # 为演示，减少数量
    "TEST_SET_SIZE": 0.2,
    "RANDOM_STATE": 42,
}

# 3. trainer (训练器) 配置
TRAINER_CONFIG = {
    "RANDOM_STATE": 42,
}

# 4. exporter (导出器) 和 verifier (验证器) 一般无需额外配置