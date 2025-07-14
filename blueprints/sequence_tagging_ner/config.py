# blueprints/sequence_tagging_ner/config.py
from pathlib import Path

# --- 基础路径配置 ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

# --- 蓝图任务定义 ---
TASKS = {
    "ner_core_entity": {
        "constitution_func_name": "NER_CORE_ENTITY_CONSTITUTION",
        "tag_map": {
            "O": 0, "B-PROJ": 1, "I-PROJ": 2,
            "B-PERS": 3, "I-PERS": 4, "B-EVT": 5, "I-EVT": 6
        },
        "expected_keys": {"tokens", "tags"}, # <--- 为新蓝图添加期望的数据键
    },
}

# --- 引擎各阶段详细配置 (生产级) ---
GENERATOR_CONFIG = {
    "MODEL_NAME": "gemini-2.0-flash",
    "TARGET_SAMPLES_PER_TASK": 5000,
    "SAMPLES_PER_REQUEST": 20,
}
REFINER_CONFIG = {
    "DEDUPE_THRESHOLD": 0.95,
    "DEDUPE_NUM_PERM": 128,
    "ANALYZE_NER_STATISTICS": True,
    "TEST_SET_SIZE": 0.2,
    "RANDOM_STATE": 42,
}
TRAINER_CONFIG = {
    "RANDOM_STATE": 42,
    "EMBEDDING_DIM": 128,
    "HIDDEN_DIM": 256,
    "DROPOUT": 0.5,
    "EPOCHS": 15,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.0015,
}