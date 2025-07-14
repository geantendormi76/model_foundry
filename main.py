# main.py
import argparse
import importlib
import sys
from pathlib import Path

# 确保项目根目录在sys.path中，以便动态导入
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from foundry_engine import data_generator, data_refiner, exporter
from foundry_engine.trainers import tfidf_classifier_trainer

def main():
    """模型铸造厂的主控制脚本。"""
    parser = argparse.ArgumentParser(
        description="zhzAI 微模型铸造厂 - 自动化生产线",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--blueprint",
        required=True,
        choices=["text_classification", "sequence_tagging_ner"],
        help="指定要构建的模型蓝图。"
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="指定要执行的流水线步骤，以逗号分隔。\n"
             "可用步骤: generate, refine, train, export\n"
             "示例: --steps=generate,refine\n"
             "默认为 'all'，执行所有步骤。"
    )
    args = parser.parse_args()

    # --- 1. 加载蓝图 ---
    print(f"=================================================")
    print(f"  铸造任务启动: 使用蓝图 '{args.blueprint}'")
    print(f"=================================================")
    try:
        config_module = importlib.import_module(f"blueprints.{args.blueprint}.config")
        constitution_module = importlib.import_module(f"blueprints.{args.blueprint}.constitution")
    except ImportError as e:
        print(f"[致命错误] 无法加载蓝图 '{args.blueprint}'。请确保相关文件存在。错误: {e}")
        sys.exit(1)

    # ======================================================================
    # === 编译器指示修正 (V2) ===
    # 错误原因: 之前的步骤列表遗漏了 'export'。
    # 解决方案: 将 'export' 添加到默认的 'all' 流程中。
    all_steps = ["generate", "refine", "train", "export"]
    # ======================================================================
    steps_to_run = args.steps.split(',') if args.steps != "all" else all_steps

    # --- 2. 按顺序执行引擎模块 ---
    if "generate" in steps_to_run:
        print("\n--- [阶段 1/4] 执行数据生成引擎 ---")
        data_generator.run(config_module, constitution_module)

    if "refine" in steps_to_run:
        print("\n--- [阶段 2/4] 执行数据精炼引擎 ---")
        data_refiner.run(config_module)

    if "train" in steps_to_run:
        print("\n--- [阶段 3/4] 执行训练引擎 ---")
        if args.blueprint == "text_classification":
            tfidf_classifier_trainer.run(config_module)
        elif args.blueprint == "sequence_tagging_ner":
            print("[提示] BiLSTM-CRF 训练器尚未实现。跳过此步骤。")
        else:
            print(f"[警告] 未知蓝图 '{args.blueprint}' 的训练器。跳过此步骤。")
            
    # ======================================================================
    # === 编译器指示修正 (V2) ===
    # 错误原因: 缺少对 exporter 模块的调用。
    # 解决方案: 添加调用 exporter.run 的逻辑块。
    if "export" in steps_to_run:
        print("\n--- [阶段 4/4] 执行预处理器导出引擎 ---")
        exporter.run(config_module)
    # ======================================================================

    print(f"\n=================================================")
    print(f"  蓝图 '{args.blueprint}' 的铸造任务已完成！")
    print(f"=================================================")

if __name__ == "__main__":
    main()