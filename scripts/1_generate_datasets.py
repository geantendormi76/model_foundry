# /home/zhz/zhzai/model_foundry/scripts/1_generate_datasets.py

import os
import asyncio
import json
import time
import random
import itertools
from typing import List, Dict, Any
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv


# 加载 .env 文件中的环境变量
load_dotenv()

# --- 配置区 ---

# 从环境变量安全地加载 API Keys
api_keys_str = os.getenv("GEMINI_API_KEYS")
if not api_keys_str:
    raise ValueError("错误：未找到 GEMINI_API_KEYS 环境变量。请在项目根目录创建 .env 文件并按格式 'GEMINI_API_KEYS=key1,key2' 设置您的 API 密钥。")

# 将逗号分隔的字符串解析为脚本所需的字典结构
API_KEYS = {key.strip(): {'status': 'ready', 'unlock_time': 0} for key in api_keys_str.split(',') if key.strip()}

if not API_KEYS:
    raise ValueError("错误：GEMINI_API_KEYS 变量为空或格式不正确。")

print(f"成功从 .env 文件加载 {len(API_KEYS)} 个 API Key。")

# 创建一个无限循环的Key迭代器
key_cycler = itertools.cycle(API_KEYS)

# Gemini 1.5 Flash 模型 (注意：Gemini 2.0 Flash 是一个假设的模型名，目前最新的是1.5系列)
MODEL_NAME = "gemini-2.0-flash"

# API 速率限制 (免费层)
REQUESTS_PER_MINUTE = 15

# 数据生成目标
TARGET_SAMPLES_PER_TASK = 5000  # 每个任务的目标样本数
SAMPLES_PER_REQUEST = 25      # 每次API请求生成多少个样本

# --- Prompt 设计区 (全中文优化版) ---

# 任务1: IsQuestionClassifier 的Prompt
IS_QUESTION_PROMPT = f"""
你是一位为机器学习分类器生成训练数据的专家。
你的任务是为个人AI助手生成一系列多样化的用户输入文本。
对于每一条文本，你必须为其分配一个分类标签：“Question”（问题）或“Statement”（陈述）。

- “Question”标签：适用于任何明确或隐含的提问。包括以“什么”、“怎么”、“谁”、“何时”、“哪里”、“为什么”开头的句子，或以问号结尾的句子。也包括那些暗示了问题的短语，例如“给我讲讲...”、“查一下...”。
- “Statement”标签：适用于任何陈述句、命令、笔记或个人想法。包括像“帮我记一下...”、“提醒我...”这样的命令，或者像“今天天气不错”这样的简单事实。

**核心指令:** 
1.  **精确生成 {SAMPLES_PER_REQUEST} 条**独一无二的示例。
2.  输出**必须是**一个严格合法的JSON数组，数组中的每个对象都包含 "text" 和 "label" 两个键。
3.  除了这个JSON数组，**不要包含任何**额外的解释、Markdown标记或其他任何文字。
4.  确保“Question”和“Statement”两种标签的数量大致均衡。
5.  所有生成的文本都必须是**中文简体**。

**输出格式示例:**
[
  {{"text": "明天下午有什么安排？", "label": "Question"}},
  {{"text": "帮我记一下，项目Alpha的核心技术是分布式图计算。", "label": "Statement"}},
  ...
]
"""

# 任务2: ConfirmationClassifier 的Prompt
CONFIRMATION_PROMPT = f"""
你是一位为机器学习分类器生成训练数据的专家。
场景是：一个AI助手向用户提出了一个确认请求（例如：“您确定要删除这条记忆吗？”）。
你的任务是生成一系列用户可能会如何回复的、多样化的例子。
对于每一条回复，你必须为其分配一个分类标签：“Affirm”（肯定）或“Deny”（否定）。

- “Affirm”标签：用户表示同意、确认或肯定的回复。这也包括那些在确认的同时提供了新信息的回复，例如：“是的，把它改成明天”。
- “Deny”标签：用户表示不同意、取消、否定的回复，或者给出了一个完全不相关的答复。

**核心指令:**
1.  **精确生成 {SAMPLES_PER_REQUEST} 条**独一无二的示例。
2.  输出**必须是**一个严格合法的JSON数组，数组中的每个对象都包含 "text" 和 "label" 两个键。
3.  除了这个JSON数组，**不要包含任何**额外的解释、Markdown标记或其他任何文字。
4.  确保“Affirm”和“Deny”两种标签的数量大致均衡。
5.  所有生成的文本都必须是**中文简体**。

**输出格式示例:**
[
  {{"text": "是的，继续吧", "label": "Affirm"}},
  {{"text": "先别，我再想想", "label": "Deny"}},
  {{"text": "改成下周五", "label": "Affirm"}},
  ...
]
"""

# --- 核心逻辑 (全新重构) ---

generation_config = genai.GenerationConfig(
    temperature=1.0, top_p=0.95, top_k=64, response_mime_type="application/json",
)


async def fetch_batch(key: str, prompt: str) -> List[Dict] | None:
    """使用指定的Key执行一次API请求"""
    try:
        # 为当前任务配置genai客户端
        genai.configure(api_key=key)
        model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)
        
        response = await model.generate_content_async(prompt)
        
        data = json.loads(response.text)
        if not isinstance(data, list):
            print(f"\n[警告] Key ...{key[-4:]} 返回了非列表数据。")
            return None
            
        return [item for item in data if isinstance(item, dict) and "text" in item and "label" in item]

    except Exception as e:
        # 将错误向上抛出，由调度器处理
        raise e

async def worker(task_name: str, prompt: str, output_file: str, pbar: tqdm):
    """一个工作单元，负责不断请求直到成功获取一个批次的数据"""
    while True:
        # 从Key池中获取一个可用的Key
        key_to_use = None
        while key_to_use is None:
            for key, state in API_KEYS.items():
                if state['status'] == 'ready':
                    key_to_use = key
                    break
            if key_to_use is None:
                # 所有Key都在冷却，等待最短的冷却时间
                await asyncio.sleep(1)
                # 检查是否有Key已解锁
                for key, state in API_KEYS.items():
                    if state['status'] == 'cooldown' and time.time() > state['unlock_time']:
                        API_KEYS[key]['status'] = 'ready'
                        print(f"\n[信息] Key ...{key[-4:]} 已冷却完毕，恢复就绪。")
        
        try:
            # 标记Key为使用中，防止其他worker同时使用
            API_KEYS[key_to_use]['status'] = 'in_use'
            
            batch_data = await fetch_batch(key_to_use, prompt)
            
            if batch_data:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for item in batch_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                pbar.update(len(batch_data))
                API_KEYS[key_to_use]['status'] = 'ready' # 成功后释放Key
                return # 工作单元完成任务，退出循环
        
        except Exception as e:
            key_display = key_to_use[-4:]
            if "429" in str(e) and "quota" in str(e).lower():
                # 【V5核心】智能熔断与冷却
                cooldown_seconds = 61 # 冷却超过1分钟，等待下一个配额周期
                unlock_time = time.time() + cooldown_seconds
                API_KEYS[key_to_use]['status'] = 'cooldown'
                API_KEYS[key_to_use]['unlock_time'] = unlock_time
                print(f"\n[严重] Key ...{key_display} 触发速率限制！进入冷却状态 {cooldown_seconds} 秒...")
            else:
                print(f"\n[错误] Key ...{key_display} 发生未知错误: {e}。将短暂冷却后重试。")
                API_KEYS[key_to_use]['status'] = 'cooldown'
                API_KEYS[key_to_use]['unlock_time'] = time.time() + 5
        
        # 无论成功失败，这个worker的本次尝试都结束了，让出控制权给调度循环

async def generate_data_for_task(task_name: str, prompt: str, target_samples: int, output_file: str):
    print(f"\n--- 开始为任务生成数据: {task_name} ---")
    
    existing_samples = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_samples = sum(1 for _ in f)
    print(f"在 {output_file} 中发现 {existing_samples} 条已存在的样本。")

    if existing_samples >= target_samples:
        print(f"已达到 {target_samples} 的样本目标。跳过此任务。")
        return

    samples_to_generate = target_samples - existing_samples
    
    with tqdm(total=samples_to_generate, desc=f"正在生成: {task_name}") as pbar:
        # 计算需要多少个工作任务（每个任务获取一个批次）
        tasks_needed = (samples_to_generate + SAMPLES_PER_REQUEST - 1) // SAMPLES_PER_REQUEST
        
        # 创建所有工作任务
        worker_tasks = [worker(task_name, prompt, output_file, pbar) for _ in range(tasks_needed)]
        
        # 并发执行
        await asyncio.gather(*worker_tasks)

async def main():
    os.makedirs("../datasets/raw", exist_ok=True)
    tasks_to_run = [
        {
            "task_name": "IsQuestionClassifier (问题/陈述分类器)", # 'name' -> 'task_name'
            "prompt": IS_QUESTION_PROMPT,
            "target_samples": TARGET_SAMPLES_PER_TASK,      # 'target' -> 'target_samples'
            "output_file": "../datasets/raw/is_question_dataset.jsonl" # 'output' -> 'output_file'
        },
        {
            "task_name": "ConfirmationClassifier (肯定/否定分类器)",
            "prompt": CONFIRMATION_PROMPT,
            "target_samples": TARGET_SAMPLES_PER_TASK,
            "output_file": "../datasets/raw/confirmation_dataset.jsonl"
        }
    ]
    start_time = time.time()
    for task_info in tasks_to_run:
        await generate_data_for_task(**task_info)
    end_time = time.time()
    print(f"\n总生成时间: {end_time - start_time:.2f} 秒。")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())