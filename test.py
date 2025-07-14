# test.py
import argparse
import importlib
import sys
from pathlib import Path

def main():
    """模型测试的统一入口。"""
    parser = argparse.ArgumentParser(description="zhzAI 微模型唤醒实验室")
    parser.add_argument(
        "--blueprint",
        required=True,
        choices=["text_classification", "sequence_tagging_ner"],
        help="指定要测试的模型蓝图。"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    
    print("="*50)
    print("  欢迎来到zhzAI“模型唤醒实验室”！")
    print(f"  正在唤醒蓝图: '{args.blueprint}'")
    print("="*50)

    try:
        # 动态导入所选蓝图的tester模块
        tester_module = importlib.import_module(f"blueprints.{args.blueprint}.tester")
        # 调用该模块的run函数
        tester_module.run(models_dir)
    except ImportError:
        print(f"[致命错误] 无法找到蓝图 '{args.blueprint}' 的测试器。")
        print(f"请确保 'blueprints/{args.blueprint}/tester.py' 文件存在。")
        sys.exit(1)
    except Exception as e:
        print(f"[致命错误] 测试过程中发生未知错误: {e}")
        sys.exit(1)
        
    print(f"\n蓝图 '{args.blueprint}' 测试结束。")

if __name__ == "__main__":
    main()