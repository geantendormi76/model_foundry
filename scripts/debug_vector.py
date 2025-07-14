# /model_foundry/scripts/debug_preprocessor.py
import joblib
from pathlib import Path

def debug_vector(task_name: str, text_input: str, project_root: Path):
    """
    加载指定任务的预处理器，并打印单个输入的稀疏向量表示。
    """
    print(f"--- 正在调试 '{task_name}' 预处理器，输入: '{text_input}' ---")
    
    models_dir = project_root / "model_foundry" / "models"
    preprocessor_path = models_dir / f"{task_name}_preprocessor.joblib"

    if not preprocessor_path.exists():
        print(f"[错误] 找不到预处理器文件: {preprocessor_path}")
        return

    try:
        pipeline = joblib.load(preprocessor_path)
        
        # 使用预处理器转换输入文本
        vector = pipeline.transform([text_input])
        
        # 打印稀疏向量的详细信息
        print(f"向量形状: {vector.shape}")
        print("非零元素的 (索引, 值):")
        # coo_matrix是稀疏矩阵的一种表示法
        cx = vector.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            # 我们只关心索引和值
            print(f"  (索引: {j}, 值: {v:.6f})")
        
        if cx.nnz == 0:
            print("  (无非零元素，这是一个零向量!)")

    except Exception as e:
        print(f"[严重错误] 调试过程中发生错误: {e}")

def main():
    # 注意：这里的路径可能需要根据您的项目结构微调
    project_root = Path(__file__).resolve().parent.parent.parent
    print(f"项目根目录检测为: {project_root}")
    
    # 我们要调试的目标
    task = "confirmation"
    inputs_to_test = ["是的", "不", "ok", "OK", "Yes"]
    
    for text in inputs_to_test:
        debug_vector(task, text, project_root)
        print("-" * 20)

if __name__ == "__main__":
    main()